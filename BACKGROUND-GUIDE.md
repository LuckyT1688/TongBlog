# 背景图片更换指南

## 当前背景配置

网站使用的是**渐变绿色背景**，定义在 `src/components/base/Background.astro` 文件中。

## 方式一：使用渐变色背景（当前方案）

### 修改渐变颜色

编辑 `src/components/base/Background.astro` 文件：

```css
/* 浅色模式背景 */
html::before {
  background: linear-gradient(
    135deg,
    #e8f5e9 0%,      /* 起始颜色 */
    #c8e6c9 25%,     /* 第二个颜色 */
    #a5d6a7 50%,     /* 中间颜色 */
    #81c784 75%,     /* 第四个颜色 */
    #66bb6a 100%     /* 结束颜色 */
  );
}

/* 深色模式背景 */
html::after {
  background: linear-gradient(
    135deg,
    #1b5e20 0%,
    #2e7d32 25%,
    #388e3c 50%,
    #43a047 75%,
    #4caf50 100%
  );
}
```

### 调整渐变方向

- `135deg` - 从左上到右下（当前设置）
- `90deg` - 从上到下
- `180deg` - 从左到右
- `45deg` - 对角线

## 方式二：使用图片背景

### 1. 准备图片

将背景图片放在以下位置：
- 浅色模式图片：`src/assets/backgrounds/light-bg.jpg`
- 深色模式图片：`src/assets/backgrounds/dark-bg.jpg`

### 2. 修改代码

编辑 `src/components/base/Background.astro`：

```html
<!-- 图片背景方案 -->
<style>
  html::before,
  html::after {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    z-index: -1;
    transition: opacity 0.5s ease-in-out;
    pointer-events: none;
  }

  html::before {
    background-image: url(/src/assets/backgrounds/light-bg.jpg);
    opacity: 1;
  }

  html.dark::before {
    opacity: 0;
  }

  html::after {
    background-image: url(/src/assets/backgrounds/dark-bg.jpg);
    opacity: 0;
  }

  html.dark::after {
    opacity: 1;
  }
</style>
```

## 方式三：混合模式（图片+渐变）

```css
html::before {
  background: 
    linear-gradient(135deg, rgba(232,245,233,0.8), rgba(102,187,106,0.8)),
    url(/src/assets/backgrounds/pattern.jpg);
  background-size: cover;
  background-position: center;
}
```

## 推荐的背景图片

### 免费图片资源网站
- **Unsplash**: https://unsplash.com/ （高质量免费图片）
- **Pexels**: https://www.pexels.com/ （免费素材）
- **Pixabay**: https://pixabay.com/ （免费图片和插画）

### 搜索关键词建议
- `abstract green background`
- `minimal gradient`
- `data visualization background`
- `tech pattern`
- `geometric shapes`

### 图片规格建议
- **分辨率**：至少 1920x1080（Full HD）
- **格式**：JPG（小文件）或 WebP（更小）
- **大小**：尽量控制在 500KB 以内

## 提交更改

修改完成后，推送到 GitHub：

```bash
git add .
git commit -m "更新背景"
git push
```

Cloudflare 会自动检测并部署新版本！

---

💡 **提示**：如果想要纯色背景，直接设置单一颜色即可，例如：
```css
html::before {
  background: #e8f5e9;  /* 浅绿色 */
}
```
