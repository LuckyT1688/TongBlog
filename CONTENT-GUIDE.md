# 内容创作指南

欢迎使用晏秋二十三的博客系统！本指南将告诉你如何在项目中创建和管理内容。

## 📝 内容目录结构

```
src/content/
├── blog/          # 博客文章（主要内容区）
├── notes/         # 动态/笔记（类似微博的短内容）
├── about/         # 关于页面
├── pages/         # 自定义页面
├── home/          # 首页配置
└── social/        # 社交链接配置
```

## ✍️ 如何创建博客文章

### 1. 在哪里创建？

**在 `src/content/blog/` 目录下创建新的 `.md` 或 `.mdx` 文件**

例如：
```
src/content/blog/my-first-article.md
```

### 2. 文章模板

每篇文章必须包含 Front Matter（文件开头的元数据）：

```markdown
---
title: "文章标题"
description: "文章简介，会显示在列表页和 SEO 中"
publishedAt: 2025-11-14
categories: ["数据分析"]  # 分类
tags: ["Python", "Pandas", "金融数据"]  # 标签
featured: true  # 是否为特色文章
status: published  # published | draft | archived
draft: false  # 是否为草稿
---

# 正文开始

这里是你的文章内容...

## 二级标题

### 三级标题

```

### 3. 完整示例

创建文件：`src/content/blog/example-article.md`

```markdown
---
title: "Python 数据分析入门"
description: "介绍 Python 数据分析的基础知识和常用库"
publishedAt: 2025-11-14
categories: ["数据分析"]
tags: ["Python", "NumPy", "Pandas"]
featured: false
status: published
draft: false
---

# Python 数据分析入门

本文将介绍 Python 数据分析的基础知识。

## 安装必要的库

\`\`\`bash
pip install numpy pandas matplotlib
\`\`\`

## 基础示例

\`\`\`python
import pandas as pd
import numpy as np

# 创建数据
data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})

print(data)
\`\`\`

## 总结

本文介绍了...
```

### 4. Front Matter 字段说明

| 字段 | 类型 | 必填 | 说明 |
|-----|------|-----|------|
| `title` | string | ✅ | 文章标题 |
| `description` | string | ✅ | 文章描述 |
| `publishedAt` | date | ✅ | 发布日期 (YYYY-MM-DD) |
| `categories` | array | ⚠️ | 分类数组 |
| `tags` | array | ⚠️ | 标签数组 |
| `status` | enum | ⚠️ | published / draft / archived |
| `featured` | boolean | ❌ | 是否特色文章，默认 false |
| `draft` | boolean | ❌ | 是否草稿，默认 false |
| `image` | string | ❌ | 封面图片路径 |

## 📌 如何创建笔记/动态

### 在 `src/content/notes/` 目录创建

笔记是较短的内容，类似微博或推文。

**示例：** `src/content/notes/today-thought.md`

```markdown
---
title: "今天的想法"
publishedAt: 2025-11-14
draft: false
---

今天学习了新的数据可视化技巧，分享一下心得...

可以添加代码：

\`\`\`python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
\`\`\`
```

## 🏠 首页配置

### 编辑 `src/content/home/-index.md`

```markdown
---
title: 首页
description: 晏秋二十三的技术博客
---

欢迎访问我的博客！
```

## 📄 自定义页面

### 在 `src/content/pages/` 创建新页面

例如：`src/content/pages/tools.md`

```markdown
---
title: "常用工具"
description: "我的常用工具推荐"
draft: false
---

# 常用工具推荐

## 数据分析工具
- Jupyter Notebook
- VS Code

## Python 库
- Pandas
- NumPy
```

然后在 `src/pages/` 创建对应的路由文件（需要一定的 Astro 知识）。

## 📷 图片管理

### 1. 存放位置

推荐将图片放在：
```
public/assets/uploads/
```

### 2. 在文章中引用

```markdown
![图片描述](/assets/uploads/my-image.jpg)
```

或在 Front Matter 中设置封面：

```yaml
---
image: "/assets/uploads/cover.jpg"
---
```

## 🏷️ 分类和标签

### 推荐的金融数据分析分类

```yaml
categories: ["数据分析", "量化投资", "风险管理", "数据可视化", "技术分享"]
```

### 推荐的标签

```yaml
tags: ["Python", "Pandas", "NumPy", "Matplotlib", "Plotly", 
       "量化策略", "回测", "VaR", "夏普比率", "机器学习"]
```

## 📊 Markdown 语法支持

### 代码块

\`\`\`python
def hello():
    print("Hello World")
\`\`\`

### 表格

| 列1 | 列2 |
|-----|-----|
| 数据1 | 数据2 |

### 数学公式（KaTeX）

行内公式：$E = mc^2$

块级公式：
$$
\text{Sharpe Ratio} = \frac{\bar{R} - R_f}{\sigma}
$$

### 引用

> 这是一段引用文字

### 列表

- 无序列表项 1
- 无序列表项 2

1. 有序列表项 1
2. 有序列表项 2

## 🚀 发布流程

1. **创建文件**：在 `src/content/blog/` 创建 `.md` 文件
2. **编写内容**：填写 Front Matter 和正文
3. **预览**：运行 `npm run dev`，访问 http://localhost:4321/blog
4. **发布**：设置 `status: published` 和 `draft: false`
5. **构建**：运行 `npm run build`
6. **部署**：上传 `dist/` 目录到服务器

## 📋 草稿管理

### 创建草稿

```yaml
---
title: "未完成的文章"
draft: true
status: draft
---
```

### 查看草稿

访问：http://localhost:4321/drafts

## 🔍 内容展示位置

| 内容类型 | 存放目录 | 展示页面 |
|---------|---------|---------|
| 博客文章 | `src/content/blog/` | `/blog` 列表页和 `/blog/文章名` 详情页 |
| 笔记动态 | `src/content/notes/` | `/notes` 或首页动态区域 |
| 关于页面 | `src/content/about/` | `/about` |

## ⚡ 快速开始示例

### 创建你的第一篇文章

1. 在 VS Code 中打开项目
2. 右键 `src/content/blog/` 文件夹
3. 选择"新建文件"
4. 命名为 `my-first-post.md`
5. 复制以下内容：

```markdown
---
title: "我的第一篇博客"
description: "这是我在晏秋二十三博客系统中的第一篇文章"
publishedAt: 2025-11-14
categories: ["技术分享"]
tags: ["博客", "开始"]
status: published
draft: false
---

# 我的第一篇博客

欢迎来到我的博客！这是我的第一篇文章。

## 关于我

我是一名金融数据分析师...

## 我的目标

在这个博客中，我将分享：

- 数据分析技巧
- Python 编程经验
- 量化投资心得

期待与大家交流！
```

6. 保存文件
7. 运行 `npm run dev`
8. 访问 http://localhost:4321/blog 查看效果

## 🛠️ 常见问题

### Q: 文章不显示？
A: 检查：
- `draft: false`
- `status: published`
- Front Matter 格式正确
- 文件在 `src/content/blog/` 目录

### Q: 图片不显示？
A: 确保：
- 图片在 `public/` 目录下
- 路径以 `/` 开头，如 `/assets/uploads/image.jpg`

### Q: 日期格式？
A: 使用 `YYYY-MM-DD` 格式，如 `2025-11-14`

### Q: 中文标题可以吗？
A: 可以！系统会自动处理 URL

## 📞 需要帮助？

如果遇到问题，可以：
1. 查看已有文章的格式作为参考
2. 检查浏览器控制台的错误信息
3. 查看终端运行日志

---

**祝你创作愉快！🎉**

晏秋二十三 | https://litong.asia
