#解决train oot分布不一致的问题

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve

def calculate_ks(y_true, y_pred):
    """计算KS值"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return np.max(tpr - fpr)

def min_bin_balanced_rebuild(train_df, oot_df, features, target, n_bins=5):
    """
    以最小分箱样本量为基准的重建方法：
    1. 找出样本量最少的分箱
    2. 所有分箱按该最小量抽取
    3. 分箱内严格保持OOT的坏样本率
    """
    # === 1. 训练OOT模型 ===
    print("训练OOT模型...")
    oot_params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'verbosity': -1
    }
    oot_data = lgb.Dataset(oot_df[features], label=oot_df[target])
    oot_model = lgb.train(oot_params, oot_data, num_boost_round=100)

    # === 2. 数据分箱 ===
    print("数据分箱处理...")
    train_df['score'] = oot_model.predict(train_df[features])
    oot_df['score'] = oot_model.predict(oot_df[features])
    
    # 使用OOT数据确定分箱边界
    bins = pd.qcut(oot_df['score'], q=n_bins, retbins=True)[1]
    bins[0], bins[-1] = -np.inf, np.inf
    
    train_df['bin'] = pd.cut(train_df['score'], bins=bins)
    oot_df['bin'] = pd.cut(oot_df['score'], bins=bins)

    # === 3. 计算OOT各分箱的坏样本率 ===
    print("计算OOT分箱统计...")
    oot_stats = oot_df.groupby('bin').agg({
        target: ['count', 'mean']
    }).reset_index()
    oot_stats.columns = ['bin', 'oot_count', 'oot_bad_rate']

    # === 4. 确定最小分箱样本量 ===
    print("\n确定基准样本量...")
    bin_sample_counts = train_df.groupby('bin').size()
    min_samples = bin_sample_counts.min()
    print(f"最小分箱样本量: {min_samples} (分箱 {bin_sample_counts.idxmin()})")

    # === 5. 按最小量抽取 + 控制坏样本率 ===
    print("\n执行基准抽样...")
    new_train_samples = []
    
    for bin_range in oot_stats['bin']:
        # 获取当前分箱数据
        train_in_bin = train_df[train_df['bin'] == bin_range]
        if len(train_in_bin) == 0:
            continue
            
        # 获取OOT的坏样本率
        target_bad_rate = oot_stats.loc[oot_stats['bin'] == bin_range, 'oot_bad_rate'].values[0]
        
        # 计算需要的好/坏样本量
        bad_samples = int(min_samples * target_bad_rate)
        good_samples = min_samples - bad_samples
        
        # 分离好坏样本
        good_data = train_in_bin[train_in_bin[target] == 0]
        bad_data = train_in_bin[train_in_bin[target] == 1]
        
        # 抽样（如果某类样本不足，则全部取用）
        sampled_good = good_data.sample(
            min(good_samples, len(good_data)), 
            random_state=42
        ) if good_samples > 0 else pd.DataFrame()
        
        sampled_bad = bad_data.sample(
            min(bad_samples, len(bad_data)), 
            random_state=42
        ) if bad_samples > 0 else pd.DataFrame()
        
        new_train_samples.append(pd.concat([sampled_good, sampled_bad]))

    # === 6. 合并结果 ===
    new_train = pd.concat(new_train_samples)
    print(f"\n原始训练集: {len(train_df)} | 新训练集: {len(new_train)}")
    
    # === 7. 分布验证 ===
    print("\n=== 分布验证 ===")
    # 样本量分布
    bin_counts = new_train['bin'].value_counts().sort_index()
    print("\n各分箱样本量:")
    for bin_range, count in bin_counts.items():
        print(f"分箱 {bin_range}: {count}样本")
    
    # 坏样本率验证
    print("\n坏样本率对比:")
    for bin_range in oot_stats['bin']:
        oot_br = oot_stats.loc[oot_stats['bin'] == bin_range, 'oot_bad_rate'].values[0]
        new_br = new_train[new_train['bin'] == bin_range][target].mean() if bin_range in bin_counts else 0
        print(f"分箱 {bin_range}: OOT {oot_br:.2%} -> 新训练集 {new_br:.2%}")
    
    return new_train.drop(['score', 'bin'], axis=1)

# ==================== 使用示例 ====================

    # 重构训练集
    new_train = min_bin_balanced_rebuild(
        train_df, oot_df,
        features=['income', 'credit_score'],
        target='target',
        n_bins=100
    )
    
