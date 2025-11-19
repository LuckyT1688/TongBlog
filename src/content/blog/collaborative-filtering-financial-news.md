---
title: 金融资讯推荐系统中的协同过滤算法实践
description: 业务需求来了，用户说资讯太多看不过来。产品说要做个性化推荐，让用户只看感兴趣的内容
publishedAt: 2025-11-19T08:25:16
updatedAt: 2025-11-19T08:25:16
status: published
featured: false
categories:
  - 数据分析
tags:
  - 推荐算法
  - 协同过滤
  - Python
  - 金融科技
draft: false
---

最近在做一个金融资讯推荐的项目，用户行为数据有了，就想着能不能根据用户的阅读习惯，给他推送些可能感兴趣的内容。自然就想到了协同过滤这个老朋友。

本文不打算讲太多理论公式，主要记录下实际操作中遇到的问题和解决思路。如果你也在做类似的推荐系统，希望这些经验能帮到你。

## 为什么选协同过滤

一开始其实考虑过几个方案：

1. **基于内容的推荐**：分析新闻标题、正文，提取关键词，找相似的推荐。问题是金融新闻的专业词汇太多，分词效果不理想，而且"相似"不代表用户想看。

2. **热门推荐**：直接推阅读量最高的。这个最简单，但对老用户体验不好，总是那几条新闻。

3. **协同过滤**：看用户行为，找到"口味相似"的用户，推荐他们看过的内容。

最后选了协同过滤，主要是因为数据现成的——用户的阅读、收藏、分享记录都有，不需要额外标注。

## 数据准备

我们的数据大概长这样：

```python
import pandas as pd
import numpy as np

# 用户行为数据
# user_id: 用户ID
# article_id: 文章ID
# action: 行为类型 (1=点击, 2=收藏, 3=分享)
# timestamp: 时间戳

data = pd.read_csv('user_behavior.csv')
print(data.head())
```

输出：
```
   user_id  article_id  action           timestamp
0     1001       20231       1  2025-11-01 09:23:15
1     1002       20231       2  2025-11-01 10:15:32
2     1001       20245       1  2025-11-01 11:04:22
3     1003       20231       1  2025-11-01 14:30:11
4     1002       20267       3  2025-11-01 15:22:05
```

### 遇到的第一个坑：如何给行为打分

刚开始直接用点击=1分，结果推荐出来的都是标题党文章（点击高但质量不行）。后来改成了加权评分：

```python
def calculate_score(action):
    """
    根据用户行为计算评分
    点击=1, 收藏=3, 分享=5
    """
    score_map = {1: 1.0, 2: 3.0, 3: 5.0}
    return score_map.get(action, 0)

data['score'] = data['action'].apply(calculate_score)
```

这个权重是试出来的，你的业务可能需要调整。我们发现分享行为特别能体现用户兴趣，所以给了最高分。

## 构建用户-文章评分矩阵

协同过滤的核心就是这个矩阵。每行是一个用户，每列是一篇文章，值是用户对文章的评分。

```python
from scipy.sparse import csr_matrix

# 聚合同一用户对同一文章的多次行为
user_item_matrix = data.groupby(['user_id', 'article_id'])['score'].sum().unstack(fill_value=0)

print(f"矩阵大小: {user_item_matrix.shape}")
print(f"稀疏度: {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.2%}")
```

输出可能是：
```
矩阵大小: (12543, 8921)
稀疏度: 99.87%
```

99%的稀疏度，这很正常。每个用户不可能看遍所有文章，大部分位置都是0。

### 第二个坑：内存爆炸

一开始直接用 DataFrame 存矩阵，结果内存占用超过 10GB。改用稀疏矩阵后降到了 200MB：

```python
# 转成稀疏矩阵
sparse_matrix = csr_matrix(user_item_matrix.values)

# 节省内存
import sys
print(f"DataFrame 大小: {sys.getsizeof(user_item_matrix) / 1024 / 1024:.2f} MB")
print(f"稀疏矩阵大小: {(sparse_matrix.data.nbytes + sparse_matrix.indptr.nbytes + 
                      sparse_matrix.indices.nbytes) / 1024 / 1024:.2f} MB")
```

## 基于用户的协同过滤（User-Based）

思路很直观：找到和目标用户兴趣相似的其他用户，推荐他们看过但目标用户没看过的文章。

### 计算用户相似度

我试了几个相似度计算方法：

```python
from sklearn.metrics.pairwise import cosine_similarity

# 余弦相似度
user_similarity = cosine_similarity(sparse_matrix)

# 转成 DataFrame 方便查看
user_sim_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)
```

**皮尔逊相关系数**也试过，但在金融资讯这个场景下，余弦相似度效果更好。可能是因为用户评分分布比较集中，不需要去均值化。

### 生成推荐

```python
def get_user_based_recommendations(user_id, top_n=10):
    """
    基于用户的协同过滤推荐
    """
    # 找到相似用户（排除自己）
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:21]
    
    # 用户已经看过的文章
    user_articles = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    
    # 候选文章及评分
    candidates = {}
    
    for sim_user, similarity in similar_users.items():
        # 相似度太低的用户不考虑
        if similarity < 0.1:
            continue
            
        # 这个相似用户看过的文章
        sim_user_articles = user_item_matrix.loc[sim_user]
        sim_user_articles = sim_user_articles[sim_user_articles > 0]
        
        for article_id, score in sim_user_articles.items():
            if article_id not in user_articles:
                # 加权：相似度 * 评分
                candidates[article_id] = candidates.get(article_id, 0) + similarity * score
    
    # 排序取 top N
    recommendations = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return [article_id for article_id, score in recommendations]

# 测试
user_id = 1001
recs = get_user_based_recommendations(user_id)
print(f"为用户 {user_id} 推荐的文章: {recs}")
```

### 第三个坑：冷启动问题

新用户没有历史行为，算不出相似度。我的处理方式：

1. 新用户前3天，推热门文章
2. 有了5条以上行为记录，开始用协同过滤
3. 行为少于5条的，协同过滤+热门混合推荐

```python
def get_recommendations_with_fallback(user_id, top_n=10):
    """
    带降级策略的推荐
    """
    # 检查用户行为数
    user_actions = data[data['user_id'] == user_id]
    
    if len(user_actions) < 5:
        # 行为太少，用热门文章
        popular = data.groupby('article_id')['score'].sum().sort_values(ascending=False)
        return popular.head(top_n).index.tolist()
    else:
        # 使用协同过滤
        return get_user_based_recommendations(user_id, top_n)
```

## 基于物品的协同过滤（Item-Based）

User-Based 的问题是用户数量太多，计算相似度矩阵很慢。我们有1万多用户，相似度矩阵就是 10000x10000，每次都要重算。

Item-Based 是反过来，计算文章之间的相似度。文章数量相对稳定，算一次可以缓存起来。

```python
# 计算文章相似度（转置后再算）
item_similarity = cosine_similarity(sparse_matrix.T)

item_sim_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)
```

### 推荐逻辑

```python
def get_item_based_recommendations(user_id, top_n=10):
    """
    基于物品的协同过滤推荐
    """
    # 用户看过的文章及评分
    user_articles = user_item_matrix.loc[user_id]
    user_articles = user_articles[user_articles > 0]
    
    # 候选文章
    candidates = {}
    
    for article_id, user_score in user_articles.items():
        # 找到相似的文章
        similar_items = item_sim_df[article_id].sort_values(ascending=False)[1:21]
        
        for sim_article, similarity in similar_items.items():
            # 用户已经看过的跳过
            if user_item_matrix.loc[user_id, sim_article] > 0:
                continue
            
            # 加权
            candidates[sim_article] = candidates.get(sim_article, 0) + similarity * user_score
    
    # 排序
    recommendations = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return [article_id for article_id, score in recommendations]
```

### 性能对比

实测下来：

- **User-Based**：实时计算，单次推荐耗时 ~2.3秒
- **Item-Based**：预计算相似度，单次推荐耗时 ~0.15秒

差距还是挺明显的。我们最后用的是 Item-Based，每天凌晨更新一次文章相似度矩阵。

## 实际效果和优化

上线后跟踪了两周数据：

- **点击率（CTR）**：从基础的 3.2% 提升到 8.7%
- **平均停留时长**：从 45秒 提升到 1分38秒
- **收藏率**：提升了 120%

说明推荐的内容确实更符合用户兴趣了。

### 后续优化点

1. **时效性加权**：金融新闻有时效性，加了时间衰减因子
   ```python
   import datetime
   
   def time_decay(timestamp, half_life=7):
       """时间衰减函数，半衰期默认7天"""
       now = datetime.datetime.now()
       days_ago = (now - timestamp).days
       return 2 ** (-days_ago / half_life)
   
   data['score'] = data['score'] * data['timestamp'].apply(time_decay)
   ```

2. **多样性控制**：避免推荐的都是同一类型的新闻
   - 用聚类算法把文章分类
   - 保证推荐列表里至少有3个不同类别

3. **负反馈处理**：用户点了立刻退出的文章，认为是负反馈
   ```python
   # 停留时长小于5秒，评分减半
   data.loc[data['duration'] < 5, 'score'] *= 0.5
   ```

## 踩过的其他坑

### 矩阵分解

试过用 SVD（奇异值分解）来降维，理论上能缓解稀疏性问题。但实际效果不如直接用相似度的效果好，可能是因为我们的数据量还不够大。

```python
from scipy.sparse.linalg import svds

# SVD 分解
U, sigma, Vt = svds(sparse_matrix, k=50)

# 重建评分矩阵
predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
```

最后没用上，但代码留着了，等数据量上来再试试。

### 推荐去重

用户可能在不同场景下看到同一批推荐，需要做去重和位置打散。

```python
def deduplicate_recommendations(user_id, new_recs, window_hours=24):
    """
    去除最近推荐过的文章
    """
    # 从缓存中读取最近推荐过的文章
    recent_recs = redis_client.lrange(f"rec:{user_id}", 0, -1)
    recent_recs = set([int(x) for x in recent_recs])
    
    # 过滤
    filtered = [x for x in new_recs if x not in recent_recs]
    
    # 更新缓存
    if filtered:
        redis_client.lpush(f"rec:{user_id}", *filtered)
        redis_client.expire(f"rec:{user_id}", window_hours * 3600)
    
    return filtered
```

## 总结

协同过滤不是什么高深的算法，核心就是"找相似"。但要做好一个推荐系统，细节处理很重要：

1. 行为权重要根据业务调整
2. 稀疏矩阵能省不少内存
3. Item-Based 在文章数量可控时更实用
4. 冷启动问题要有降级方案
5. 时效性、多样性都要考虑

代码层面，Python 的 scikit-learn、scipy 这些库基本够用了。如果要上生产环境，考虑用 Spark MLlib 或者 TensorFlow Recommenders 来做分布式计算。

最后说句实话：推荐算法只是工具，真正决定效果的还是对业务的理解。金融资讯用户关心什么、什么时候需要什么内容，这些比算法本身更重要。

算法能做的，就是把合适的内容在合适的时候推给合适的人。仅此而已。

---

有问题欢迎留言交流。
