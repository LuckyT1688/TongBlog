---
title: CDN 完全指南：从原理到实战，国内外服务商全面对比
description: 网站慢如蜗牛？访问经常超时？看完这篇文章，你会知道 CDN 是怎么让你的网站飞起来的
publishedAt: 2025-11-19
updatedAt: 2025-11-19
status: published
featured: true
categories:
  - 运维
tags:
  - CDN
  - 网络
  - 性能优化
  - Cloudflare
draft: false
---

上线了个人博客之后，发现一个问题：深圳的朋友说访问很快，北京的朋友说加载要好几秒。后来才知道，服务器在深圳，北京访问当然慢。

这时候就需要 **CDN** 了。

这篇文章从零开始讲 CDN，包括原理、使用场景、国内外主流服务商对比，以及实战经验。

## CDN 是什么？

**CDN = Content Delivery Network（内容分发网络）**

**一句话解释：**
把你的网站内容复制到全球各地的服务器上，让用户从最近的服务器访问，大幅提升速度。

### 生活中的类比

你可以把 CDN 想象成**快递的分仓系统**：

**没有分仓（没有 CDN）：**
```
北京用户下单
    ↓
等待 2 天
    ↓
从深圳总仓发货
    ↓
收到货物（慢）
```

**有分仓（有 CDN）：**
```
北京用户下单
    ↓
从北京分仓发货
    ↓
2 小时送达（快）
```

网站访问也是一样的道理：

```
没有 CDN：
北京用户 → 2000km → 深圳服务器 → 延迟 100ms

有 CDN：
北京用户 → 5km → 北京 CDN 节点 → 延迟 5ms
```

## CDN 工作原理

### 基本流程

```
用户访问 www.example.com
         ↓
    DNS 智能解析
         ↓
   返回最近的 CDN 节点 IP
         ↓
┌──────────────────────────────┐
│ 北京节点  上海节点  深圳节点    │  ← CDN 边缘节点
└──────────────────────────────┘
         ↓
    CDN 节点检查缓存
         ↓
   有缓存？
   /      \
  是       否
  ↓        ↓
返回缓存  回源站拉取
         ↓
    你的服务器（源站）
         ↓
    CDN 缓存并返回
```

### 第一次访问（冷启动）

1. **DNS 解析**：用户访问域名，DNS 返回最近的 CDN 节点 IP
2. **请求转发**：CDN 节点没有缓存，向源站请求内容
3. **缓存存储**：CDN 节点缓存内容，并返回给用户
4. **设置过期时间**：根据配置设置缓存有效期（TTL）

### 后续访问（缓存命中）

1. **直接返回**：CDN 节点检测到有缓存
2. **秒级响应**：直接从本地返回，延迟极低
3. **节省带宽**：不需要回源站，减少源站压力

### 实际例子

假设你的博客有张图片：`https://blog.com/photo.jpg`

**没有 CDN：**
```
用户在北京访问
   ↓
请求发送到深圳服务器（物理距离 2000km）
   ↓
网络延迟：100ms
带宽限制：服务器出口带宽 10Mbps
下载速度：~1MB/s
用户体验：图片加载慢
```

**有 CDN：**
```
用户在北京访问
   ↓
DNS 解析到北京 CDN 节点
   ↓
北京节点有缓存，直接返回
   ↓
网络延迟：5ms
带宽充足：CDN 节点带宽 1Gbps
下载速度：~50MB/s
用户体验：秒开
```

## CDN 的核心概念

### 1. 边缘节点（Edge Node）

离用户最近的 CDN 服务器。

**节点分布：**
- 一线城市：北京、上海、广州、深圳
- 二线城市：杭州、成都、武汉等
- 海外：美国、欧洲、亚太等

**节点越多 = 覆盖越广 = 速度越快**

### 2. 回源（Origin Pull）

CDN 节点没有缓存时，向源站（你的服务器）请求数据。

```
回源率 = (回源请求数 / 总请求数) × 100%
```

**理想情况：**
- 回源率 < 10%
- 缓存命中率 > 90%

**回源太多的问题：**
- 源站压力大
- CDN 效果差
- 成本高（CDN 回源也计费）

### 3. 缓存命中率（Cache Hit Rate）

有多少请求直接从 CDN 返回，不需要回源。

```
命中率 = (CDN 直接返回的请求 / 总请求) × 100%
```

**影响因素：**
- 缓存时间（TTL）设置
- 内容类型（静态 vs 动态）
- 访问模式（热点内容更容易命中）

**优化方法：**
- 静态资源设置长缓存（1 个月）
- 使用文件版本号（`style.v2.css`）
- 预热热点内容

### 4. 缓存刷新（Purge）

当你更新网站内容后，需要清除 CDN 缓存。

**方式：**
1. **URL 刷新**：刷新指定文件
2. **目录刷新**：刷新整个目录
3. **全站刷新**：清空所有缓存（慎用）

**注意：**
- 刷新有次数限制（比如每天 1000 次）
- 刷新需要时间生效（5-10 分钟）

### 5. 防盗链（Hotlink Protection）

防止别人直接引用你的资源。

**场景：**
```
别人的网站：
<img src="https://你的博客.com/大图.jpg">
↓
消耗你的 CDN 流量 💸
```

**解决方案：**
```
检查 HTTP Referer
如果不是你的域名 → 返回 403 或替换为警告图
```

## CDN 能做什么？

### 1. 加速网站访问

**适用内容：**
- ✅ 图片、视频
- ✅ CSS、JavaScript
- ✅ 字体文件
- ✅ 下载文件
- ⚠️ HTML 页面（部分 CDN 支持）
- ❌ 动态 API（需要专门的动态加速）

**加速效果：**
```
优化前：首屏加载 3 秒
优化后：首屏加载 0.5 秒
```

### 2. 减少服务器压力

**对比：**
```
没有 CDN：
1 万用户 → 全部打到源站 → 服务器崩溃

有 CDN（命中率 90%）：
1 万用户 → 9000 个请求 CDN 处理
         → 1000 个请求到源站
         → 服务器轻松应对
```

### 3. 防御 DDoS 攻击

**原理：**
- 源站 IP 隐藏在 CDN 后面
- 攻击流量分散到全球节点
- CDN 自带基础防护

**Cloudflare 案例：**
- 2022 年抵御 26M rps（每秒 2600 万请求）的 DDoS 攻击
- 免费用户也享受基础防护

### 4. 节省带宽成本

**对比：**
```
没有 CDN：
服务器带宽：100Mbps = 800 元/月
月流量：5TB × 1 元/GB = 5000 元

有 CDN（缓存命中 90%）：
服务器带宽：10Mbps = 80 元/月
月流量：500GB × 1 元/GB = 500 元
CDN 流量：4.5TB × 0.2 元/GB = 900 元
总成本：1480 元/月（省 70%）
```

### 5. 提升 SEO 排名

Google 的排名因素之一是**网站速度**。

**数据：**
- 加载时间 < 1 秒：排名提升
- 加载时间 > 3 秒：跳出率 +53%

## 国内外主流 CDN 服务商

### 对比总览表

| 服务商 | 国内速度 | 海外速度 | 价格 | 免费额度 | 备案 | 推荐度 |
|--------|---------|---------|------|---------|------|--------|
| Cloudflare | ⭐⭐ | ⭐⭐⭐⭐⭐ | **免费** | **无限** | ❌ | ⭐⭐⭐⭐⭐ |
| 腾讯云 EdgeOne | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 中 | ❌ | ✅ | ⭐⭐⭐⭐ |
| 阿里云 CDN | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 中高 | ❌ | ✅ | ⭐⭐⭐⭐ |
| 火山引擎 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 低 | ✅ | ✅ | ⭐⭐⭐⭐ |
| 七牛云 | ⭐⭐⭐ | ⭐⭐ | 低 | 10GB/月 | ✅ | ⭐⭐⭐ |
| 又拍云 | ⭐⭐⭐ | ⭐⭐ | 低 | 联盟免费 | ✅ | ⭐⭐⭐ |
| AWS CloudFront | ⭐⭐ | ⭐⭐⭐⭐⭐ | 高 | 1TB/年 | ❌ | ⭐⭐⭐⭐ |

---

### 1. Cloudflare（国际首选）

**官网：** https://www.cloudflare.com

**优势：**
- ✅ **完全免费**：无限流量、无限请求
- ✅ **全球 300+ 节点**：覆盖 120+ 国家
- ✅ **免费 SSL**：自动签发证书
- ✅ **DDoS 防护**：免费提供
- ✅ **开发者友好**：API 强大，文档完善
- ✅ **Pages/Workers**：免费托管静态站点

**劣势：**
- ❌ 国内访问慢（无大陆节点）
- ❌ `.pages.dev` 域名在国内受限
- ❌ 需要自定义域名才能在国内正常访问

**价格：**
```
Free 套餐：完全免费
Pro 套餐：20 美元/月（更多功能）
```

**适合场景：**
- 个人博客、技术文档
- 海外用户为主的网站
- 不想备案的站点
- 开发者学习测试

**我的实战经验：**
```javascript
// Cloudflare Pages 部署配置
// 文件：wrangler.toml
name = "tongblog"
pages_build_output_dir = "dist"

[build]
command = "npm run build"

[[redirects]]
from = "/old-url"
to = "/new-url"
status = 301
```

---

### 2. 腾讯云 EdgeOne（国内首选）

**官网：** https://www.tencentcloud.com/products/eo

**优势：**
- ✅ **国内节点最多**：2800+ 节点
- ✅ **速度极快**：国内延迟 10-30ms
- ✅ **企业级防护**：DDoS、CC 防御
- ✅ **与腾讯云生态集成**：COS、CLB 无缝对接
- ✅ **技术成熟**：腾讯游戏、微信同款技术

**劣势：**
- ❌ 无免费额度
- ❌ 需要备案
- ❌ 海外节点较少

**价格：**
```
流量计费：0.24 元/GB 起
请求计费：0.01 元/万次
新用户：有试用额度
```

**适合场景：**
- 国内用户为主的网站
- 企业官网、电商平台
- 已在用腾讯云的项目

---

### 3. 阿里云 CDN

**官网：** https://www.aliyun.com/product/cdn

**优势：**
- ✅ **节点最多**：3200+ 节点
- ✅ **稳定性好**：阿里云基础设施
- ✅ **生态完善**：与 OSS、ECS 深度集成
- ✅ **企业认可度高**

**劣势：**
- ❌ 价格较高
- ❌ 控制台复杂
- ❌ 海外流量贵

**价格：**
```
流量计费：0.29 元/GB 起
HTTPS 请求：0.05 元/万次
```

**适合场景：**
- 大型企业
- 已在用阿里云的项目
- 对稳定性要求高的业务

---

### 4. 火山引擎 CDN（性价比之王）

**官网：** https://www.volcengine.com/products/cdn

**优势：**
- ✅ **价格便宜**：比阿里腾讯便宜 20-30%
- ✅ **抖音同款**：字节跳动技术
- ✅ **边缘计算强**
- ✅ **新用户福利**：有免费额度

**劣势：**
- ❌ 品牌认知度低
- ❌ 文档不够完善
- ❌ 客户案例少

**价格：**
```
流量计费：0.19 元/GB 起
新用户：赠送 50GB 流量包
```

**适合场景：**
- 成本敏感的项目
- 视频、直播类应用
- 想尝鲜新技术

---

### 5. 七牛云（开发者友好）

**官网：** https://www.qiniu.com/products/cdn

**优势：**
- ✅ **有免费额度**：10GB/月
- ✅ **文档清晰**：开发者友好
- ✅ **价格便宜**：0.18 元/GB
- ✅ **对象存储集成**：Kodo + CDN 一体化

**劣势：**
- ❌ 节点较少
- ❌ 稳定性一般
- ❌ 企业客户少

**价格：**
```
免费额度：10GB/月（HTTP）
流量计费：0.18 元/GB
```

**适合场景：**
- 个人博客、小站
- 开发者学习
- 流量不大的项目

---

### 6. 又拍云（联盟计划）

**官网：** https://www.upyun.com

**优势：**
- ✅ **联盟计划**：添加 Logo 可免费使用
- ✅ **价格实惠**：0.16 元/GB
- ✅ **图片处理**：图片优化、水印
- ✅ **开发者社区活跃**

**劣势：**
- ❌ 规模较小
- ❌ 节点不如大厂

**价格：**
```
联盟计划：免费（需添加 Logo）
流量计费：0.16 元/GB
```

**适合场景：**
- 个人博客（可免费）
- 图片、音视频类网站
- 预算有限的小项目

---

### 7. AWS CloudFront（海外专业）

**官网：** https://aws.amazon.com/cloudfront/

**优势：**
- ✅ **全球覆盖**：225+ 边缘位置
- ✅ **技术领先**：亚马逊云基础设施
- ✅ **企业级**：Netflix、Airbnb 在用
- ✅ **免费额度**：1TB/年

**劣势：**
- ❌ 价格贵
- ❌ 配置复杂
- ❌ 国内访问慢

**价格：**
```
免费套餐：1TB 流量/年（新用户）
流量计费：0.085 美元/GB 起
```

**适合场景：**
- 海外业务
- 已在用 AWS 的企业
- 对技术要求高的项目

---

## 如何选择 CDN？

### 决策树

```
你的网站用户主要在哪？
    ↓
├─ 国内用户为主
│   ↓
│   域名备案了吗？
│   ↓
│   ├─ 已备案 → 腾讯云 EdgeOne / 阿里云 CDN
│   │             （追求速度选腾讯，追求稳定选阿里）
│   │
│   └─ 未备案 → Cloudflare + 自定义域名
│                （速度一般，但免费）
│
├─ 海外用户为主
│   ↓
│   预算充足吗？
│   ↓
│   ├─ 充足 → AWS CloudFront
│   │          （专业，企业级）
│   │
│   └─ 有限 → Cloudflare
│              （免费，够用）
│
└─ 国内外都有
    ↓
    用双 CDN 方案
    ↓
    DNS 智能解析：
    国内 IP → 腾讯云
    海外 IP → Cloudflare
```

### 按场景推荐

**个人博客：**
- 首选：**Cloudflare**（免费）
- 备选：七牛云、又拍云（有免费额度）

**企业官网：**
- 首选：**腾讯云 EdgeOne**（速度快）
- 备选：阿里云 CDN（稳定）

**视频网站：**
- 首选：**火山引擎**（性价比高，视频加速强）
- 备选：七牛云（有直播功能）

**海外业务：**
- 首选：**Cloudflare**（免费，全球覆盖）
- 备选：AWS CloudFront（企业级）

**电商平台：**
- 首选：**阿里云 CDN**（稳定性好）
- 备选：腾讯云 EdgeOne（速度快）

---

## 实战：Cloudflare 配置指南

我的博客用的是 Cloudflare，下面分享配置过程。

### 1. 部署到 Cloudflare Pages

```bash
# 安装 Wrangler CLI
npm install -g wrangler

# 登录
wrangler login

# 部署
wrangler pages deploy dist --project-name=tongblog
```

### 2. 绑定自定义域名

**Cloudflare 控制台：**
```
Pages → 你的项目 → Custom domains
→ Add custom domain
→ 输入域名：blog.example.com
→ 自动配置 DNS
```

### 3. 配置缓存规则

**创建 Page Rule：**
```
URL 匹配：blog.example.com/images/*
设置：
  - Cache Level: Cache Everything
  - Edge Cache TTL: 1 month
  - Browser Cache TTL: 1 hour
```

### 4. 优化性能

**启用功能：**
- ✅ Auto Minify（自动压缩 CSS/JS/HTML）
- ✅ Brotli 压缩
- ✅ HTTP/3（QUIC）
- ✅ Early Hints

**配置示例：**
```javascript
// _headers 文件
/*
  Cache-Control: public, max-age=3600
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  X-XSS-Protection: 1; mode=block

/images/*
  Cache-Control: public, max-age=31536000, immutable

/api/*
  Cache-Control: no-cache
```

### 5. 刷新缓存

**方法 1：API**
```bash
curl -X POST "https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache" \
  -H "Authorization: Bearer {api_token}" \
  -H "Content-Type: application/json" \
  --data '{"files":["https://blog.example.com/style.css"]}'
```

**方法 2：控制台**
```
Caching → Configuration → Purge Cache
→ Custom Purge → 输入 URL
```

---

## 实战：国内 CDN 配置（以七牛云为例）

### 1. 创建对象存储空间

```bash
# 安装七牛云 CLI
pip install qiniu

# 配置
qshell account <AccessKey> <SecretKey> <Name>

# 上传文件
qshell qupload upload.json
```

**upload.json：**
```json
{
  "src_dir": "./dist",
  "bucket": "myblog",
  "ignore_dir": false,
  "overwrite": true,
  "check_exists": true
}
```

### 2. 配置 CDN 加速域名

```
对象存储 → 空间管理 → 域名管理
→ 添加域名：cdn.example.com
→ 配置 CNAME：cdn.example.com → xxx.qiniudns.com
```

### 3. 设置缓存规则

```
CDN 配置 → 缓存配置
→ 添加规则：
  - 文件类型：.jpg, .png, .css, .js
  - 缓存时间：30 天
  - 目录：/images/*
  - 缓存时间：90 天
```

### 4. 配置 HTTPS

```
域名管理 → SSL 证书
→ 上传证书（或免费申请）
→ 强制 HTTPS
```

---

## CDN 性能优化技巧

### 1. 合理设置缓存时间

```nginx
# 静态资源 - 长缓存
location ~* \.(jpg|jpeg|png|gif|ico|css|js|woff2)$ {
    expires 30d;
    add_header Cache-Control "public, immutable";
}

# HTML - 短缓存
location ~* \.html$ {
    expires 1h;
    add_header Cache-Control "public, must-revalidate";
}

# API - 不缓存
location /api/ {
    add_header Cache-Control "no-cache, no-store, must-revalidate";
}
```

### 2. 使用文件版本号

**方法 1：查询参数**
```html
<link rel="stylesheet" href="/style.css?v=1.2.3">
```

**方法 2：文件名哈希（推荐）**
```html
<link rel="stylesheet" href="/style.a7b3c2d1.css">
```

**Webpack 配置：**
```javascript
module.exports = {
  output: {
    filename: '[name].[contenthash].js',
    clean: true
  }
};
```

### 3. 图片优化

**使用 WebP 格式：**
```html
<picture>
  <source srcset="/image.webp" type="image/webp">
  <img src="/image.jpg" alt="图片">
</picture>
```

**CDN 自动转换（七牛云）：**
```
原图：https://cdn.example.com/photo.jpg
转 WebP：https://cdn.example.com/photo.jpg?imageView2/format/webp
```

### 4. 预热热点内容

**适用场景：**
- 新文章发布
- 活动页面上线
- 大促前准备

**七牛云预热：**
```python
from qiniu import Auth, http

# 预热 URL
def prefetch_urls(urls):
    auth = Auth(access_key, secret_key)
    url = 'http://fusion.qiniuapi.com/v2/tune/prefetch'
    body = {'urls': urls}
    
    ret, info = http._post_with_auth(url, body, auth)
    return ret
```

### 5. 监控缓存命中率

**关键指标：**
```
命中率 > 90%：优秀
命中率 70-90%：良好
命中率 < 70%：需要优化
```

**优化方向：**
- 延长缓存时间
- 预热热点资源
- 减少动态内容

---

## 常见问题与解决方案

### 1. 更新网站后，用户看到的还是旧内容

**原因：** CDN 缓存未刷新

**解决：**
```bash
# 方法 1：刷新指定 URL
# Cloudflare
curl -X POST "https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache" \
  -H "Authorization: Bearer {token}" \
  --data '{"files":["https://blog.com/index.html"]}'

# 方法 2：使用版本号（推荐）
# 修改文件名：style.v2.css
```

### 2. CDN 流量异常暴增

**可能原因：**
- 被盗链
- DDoS 攻击
- 热点资源被转发

**解决：**
```nginx
# 配置防盗链
valid_referers none blocked blog.example.com *.blog.example.com;
if ($invalid_referer) {
    return 403;
}

# Cloudflare 设置访问规则
```

### 3. HTTPS 证书错误

**原因：** CDN 与源站证书不匹配

**解决：**
```
Cloudflare：
SSL/TLS → 加密模式 → Full (strict)

确保源站有有效 SSL 证书
```

### 4. 某些地区访问慢

**诊断：**
```bash
# 测试不同地区延迟
curl -o /dev/null -s -w "Time: %{time_total}s\n" https://blog.com

# 使用多地域测速工具
# https://www.17ce.com
```

**解决：**
- 检查 DNS 解析
- 增加 CDN 节点覆盖
- 考虑多 CDN 方案

---

## 总结

### CDN 三个核心价值

1. **加速访问**：延迟从 100ms 降到 10ms
2. **降低成本**：节省 70% 带宽费用
3. **提升稳定性**：防御攻击，分散压力

### 选择建议

**个人博客：**
- 直接用 **Cloudflare**，免费够用

**企业网站：**
- 国内用户 → **腾讯云 EdgeOne**
- 海外用户 → **AWS CloudFront**
- 预算有限 → **火山引擎**

**最佳实践：**
- 静态资源设置长缓存
- 使用文件版本号
- 监控缓存命中率
- 定期优化性能

### 我的实战经验

用 Cloudflare Pages 部署博客后：
- ✅ 首屏加载从 3 秒降到 0.5 秒
- ✅ 服务器压力减少 90%
- ✅ 完全免费，省下每月 200 元服务器费用
- ⚠️ 国内访问速度一般（计划绑定自定义域名）

CDN 不是银弹，但对于绝大多数网站来说，是**性价比最高的性能优化手段**。

如果你的网站还没用 CDN，赶紧试试吧！

---

**参考资料：**
- Cloudflare 官方文档
- 腾讯云 EdgeOne 最佳实践
- Web Performance 权威指南
