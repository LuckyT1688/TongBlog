# Cloudflare Pages 部署指南

## 一、首次设置（只需做一次）

### 1. 创建 GitHub 仓库
```bash
# 初始化 Git（如果还没有）
git init

# 添加所有文件
git add .

# 首次提交
git commit -m "Initial commit"

# 在 GitHub 创建新仓库后，关联远程仓库
git remote add origin https://github.com/你的用户名/TongBlog.git

# 推送到 GitHub
git push -u origin main
```

### 2. 连接 Cloudflare Pages

1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. 进入 **Workers & Pages** > **Create application** > **Pages** > **Connect to Git**
3. 选择你的 GitHub 仓库 `TongBlog`
4. 配置构建设置：

   - **Framework preset**: `Astro`
   - **Build command**: `npm run build`
   - **Build output directory**: `dist`
   - **Node version**: 18 或更高

5. 点击 **Save and Deploy**

## 二、日常写博客流程（超简单！）

### 方式一：本地编写（推荐）

```bash
# 1. 在 src/content/blog/ 创建新的 .md 文件
# 例如：src/content/blog/my-new-post.md

# 2. 写完后提交
git add .
git commit -m "添加新文章：文章标题"
git push

# 3. Cloudflare 自动检测更新，自动构建部署（约 1-2 分钟）
# 4. 完成！访问你的网站即可看到新文章
```

### 方式二：GitHub 在线编辑

1. 访问 GitHub 仓库
2. 进入 `src/content/blog/`
3. 点击 **Add file** > **Create new file**
4. 写文章，提交
5. Cloudflare 自动部署

## 三、文章模板

在 `src/content/blog/` 创建 `.md` 文件，格式如下：

```markdown
---
title: "文章标题"
description: "文章简介"
date: 2025-11-14
author: "Tong"
tags: ["Python", "数据分析"]
categories: ["技术"]
draft: false
---

# 正文开始

这里是文章内容...
```

## 四、常用命令

```bash
# 本地预览
npm run dev

# 本地构建测试
npm run build

# 查看构建结果
npm run preview

# 提交并部署
git add .
git commit -m "更新内容"
git push
```

## 五、优势

✅ **无需手动构建**：推送代码后自动构建
✅ **全球 CDN**：Cloudflare 边缘网络加速
✅ **免费额度**：每月 500 次构建，无限带宽
✅ **自动 HTTPS**：免费 SSL 证书
✅ **版本控制**：Git 保存所有历史记录
✅ **随时随地**：任何能访问 GitHub 的地方都能发文章

## 六、故障排查

### 构建失败？
- 检查 Cloudflare Pages 的构建日志
- 确保 `package.json` 中的脚本正确
- Node 版本是否 >= 18

### 文章不显示？
- 检查文章 frontmatter 中 `draft: false`
- 确保文件在 `src/content/blog/` 目录
- 检查文件名和格式

---

**就是这么简单！** 每次写完文章只需 `git push`，剩下的交给 Cloudflare！
