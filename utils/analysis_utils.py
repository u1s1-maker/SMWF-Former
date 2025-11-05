# utils/analysis_utils.py
import torch
from thop import profile, clever_format

def count_parameters_detailed(model, model_name="Model", args=None, verbose=True):
    """对模型进行详细的参数分析，打印总参数量和主要模块的参数量。"""
    if model is None:
        if verbose: print(f"[ANALYSIS] 模型 {model_name} 未实例化，无法进行参数分析。")
        return {"total": 0, "trainable": 0, "modules": {}}

    if verbose:
        print(f"\n--- 详细参数分析 for: {model_name} ---")
        if args:
            print(f"  Config: seq_len={args.seq_len}, pred_len={args.pred_len}, enc_in={args.enc_in}, d_model={args.d_model}, e_layers={args.e_layers}, lifting_levels={getattr(args, 'lifting_levels', 'N/A')}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数数量: {trainable_params:,}")

    if total_params == 0 and verbose:
        print("  模型没有参数。")

    module_param_counts = {}
    if verbose and total_params > 0:
        print("\n  --- 按主要子模块划分的参数量 ---")
        for name, child_module in model.named_children():
            child_params_count = sum(p.numel() for p in child_module.parameters())
            if child_params_count > 0:
                percentage = (child_params_count / total_params) * 100
                print(f"    - {name:<30}: {child_params_count:>12,} 参数 ({percentage:.2f}%)")
                module_param_counts[name] = {
                    "params": child_params_count,
                    "percentage": percentage
                }
    if verbose: print("--- 参数分析结束 ---")
    return {
        "total": total_params,
        "trainable": trainable_params,
        "modules": module_param_counts
    }


def calculate_flops_and_params(model, dummy_inputs_func, configs, model_name="Model", verbose=True):
    """
    使用 thop 计算模型的 FLOPs 和参数量。
    dummy_inputs_func 是一个接收 configs 并返回虚拟输入的函数。
    """
    if model is None:
        if verbose: print(f"[ANALYSIS] 模型 {model_name} 未实例化，无法进行FLOPs分析。")
        return {"macs": -1, "flops": -1, "params_thop": -1}

    if verbose:
        print(f"\n--- FLOPs 分析 for: {model_name} ---")
        if configs:
             print(f"  Input Config: seq_len={configs.seq_len}, enc_in={configs.enc_in}")


    model.eval()
    dummy_inputs = dummy_inputs_func(configs)

    # 针对 KMeans 的特殊处理 (如果您的模型包含它并用于FLOPs分析)
    original_clusters = None
    kmeans_model_attr = None
    if hasattr(model, 'kmeans') and isinstance(getattr(model, 'kmeans'), object): # 简单检查
        kmeans_model_attr = model.kmeans
        if hasattr(model, 'clusters'): # 假设您的模型使用 self.clusters
            original_clusters = model.clusters
            # 为FLOPs分析设置一个虚拟的、固定大小的clusters属性
            try:
                num_channels = configs.enc_in
                n_clusters = configs.n_clusters
                if num_channels > 0 and n_clusters > 0 :
                    model.clusters = torch.zeros(num_channels, dtype=torch.long, device=next(model.parameters()).device) % n_clusters
            except Exception as e:
                if verbose: print(f"  警告: 尝试为KMeans设置虚拟clusters时出错: {e}")
    elif hasattr(model, 'trend_linear') and hasattr(model.trend_linear, 'n_clusters'): # 另一种常见模式
         if hasattr(model, 'clusters'):
            original_clusters = model.clusters
            try:
                num_channels = configs.enc_in
                n_clusters = model.trend_linear.n_clusters
                if num_channels > 0 and n_clusters > 0 :
                    model.clusters = torch.zeros(num_channels, dtype=torch.long, device=next(model.parameters()).device) % n_clusters
            except Exception as e:
                if verbose: print(f"  警告: 尝试为ClusteredLinear设置虚拟clusters时出错: {e}")


    macs, params_from_thop = -1, -1
    flops = -1
    try:
        macs, params_from_thop = profile(model, inputs=dummy_inputs, verbose=False)

        # 手动将单位转换为 G (Giga) 和 M (Mega)
        gmacs = macs / 1e9  # 1 Giga = 10^9
        flops = 2 * macs
        gflops = flops / 1e9
        m_params = params_from_thop / 1e6  # 参数通常用 M (Mega) 表示，1 Mega = 10^6

        if verbose:
            # 使用 f-string 进行格式化，保留3位小数
            print(f"  估算的 GMACs: {gmacs:.3f}G")
            print(f"  估算的 GFLOPs: {gflops:.3f}G (GMACs * 2)")
            print(f"  thop 报告的参数量: {m_params:.3f}M")

    except Exception as e:
        if verbose:
            print(f"  无法计算 FLOPs: {e}")
            print("  可能原因: thop 不支持某些自定义模块/操作，或输入形状不匹配。")
    finally:
        # 恢复 KMeans 相关属性
        if kmeans_model_attr is not None and hasattr(model, 'kmeans'):
            pass # 通常KMeans对象本身不需要恢复，主要是 self.clusters
        if original_clusters is not None and hasattr(model, 'clusters'):
            model.clusters = original_clusters
        # model.train() # 如果后续还需训练，则恢复训练模式 (通常在 Exp 类中管理模式)

    if verbose: print("--- FLOPs 分析结束 ---")
    return {"macs": macs, "flops": flops, "params_thop": params_from_thop}


def create_dummy_inputs_for_forecasting_model(configs, batch_size=1):
    """
    根据 configs 创建适用于大多数时序预测模型的虚拟输入。
    返回 (x_enc, x_mark_enc, x_dec, x_mark_dec)
    """
    seq_len = configs.seq_len
    enc_in = configs.enc_in
    pred_len = configs.pred_len
    label_len = configs.label_len
    dec_in = getattr(configs, 'dec_in', enc_in) # 确保 dec_in 存在

    # 估算时间戳特征数量
    num_time_features = 0
    if hasattr(configs, 'embed') and configs.embed == 'timeF':
        freq = getattr(configs, 'freq', 'h')
        if freq == 's': num_time_features = 6 # (year,month,day,hour,minute,second)
        elif freq == 't': num_time_features = 5 # (year,month,day,hour,minute)
        elif freq == 'h': num_time_features = 4 # (year,month,day,hour)
        elif freq == 'd': num_time_features = 3 # (year,month,day)
        elif freq in ['w', 'm']: num_time_features = 2 # (year,month) or (year,week)
        # 根据你的 DataEmbedding_inverted 实现的具体细节调整
    # 对于 'fixed' 或 'learned' 嵌入，x_mark_enc/dec 可能为 None 或有不同结构

    dummy_x_enc = torch.randn(batch_size, seq_len, enc_in)
    dummy_x_mark_enc = torch.randn(batch_size, seq_len, num_time_features) if num_time_features > 0 else None

    effective_dec_len = label_len + pred_len
    dummy_x_dec = torch.randn(batch_size, effective_dec_len, dec_in)
    dummy_x_mark_dec = torch.randn(batch_size, effective_dec_len, num_time_features) if num_time_features > 0 else None

    # 模型可能期望这些输入在特定设备上
    # device = next(model.parameters()).device # 如果模型已构建并传递进来
    # dummy_x_enc = dummy_x_enc.to(device)
    # if dummy_x_mark_enc is not None: dummy_x_mark_enc = dummy_x_mark_enc.to(device)
    # ...
    # 但这里仅创建CPU张量，在调用 calculate_flops 时模型和输入会在同一设备

    return (dummy_x_enc, dummy_x_mark_enc, dummy_x_dec, dummy_x_mark_dec)