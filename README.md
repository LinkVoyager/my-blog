# Link 的个人站点

基于 [Astro](https://astro.build/) 的个人博客与笔记 monorepo，统一承载博客文章和知识笔记。

## 站点

| 站点 | 地址 | 说明 |
|---|---|---|
| 📝 博客 | [link-blog.vercel.app](https://link-blog.vercel.app) | 技术文章、项目记录 |
| 📚 笔记 | [links-note.vercel.app](https://links-note.vercel.app) | 知识库，按主题分类 |

## 技术栈

| 站点 | 框架 | 主题 | 样式 | 部署 |
|---|---|---|---|---|
| blog | Astro 7 | [AstroPaper](https://github.com/satnaing/astro-paper) | Tailwind CSS 4 | Vercel |
| note | Astro 7 | [Starlight](https://starlight.astro.build/) | Starlight 内置 | Vercel |

## 项目结构

```
├── blog/                    # 博客站 (AstroPaper)
│   └── src/
│       ├── content/
│       │   ├── posts/       # 博客文章 (*.md)
│       │   └── pages/       # 独立页面 (关于等)
│       ├── pages/           # Astro 页面路由
│       ├── layouts/         # 页面布局
│       ├── components/      # 公共组件
│       └── i18n/lang/       # 多语言翻译 (zh-CN / en)
├── note/                    # 笔记站 (Starlight)
│   ├── astro.config.mjs     # Starlight 配置 (侧边栏、分类)
│   └── src/content/docs/    # 笔记内容 (*.mdx)
│       ├── deep-learning/   # 深度学习
│       ├── code-algorithm/  # 代码算法
│       ├── tools/           # 工具使用
│       └── minds/           # 随想记录
```

## 本地运行

```bash
# 博客站
cd blog
npm install
npm run dev        # http://localhost:4321

# 笔记站
cd note
npm install
npm run dev        # http://localhost:4322
```

## 发布文章

### 写博客

在 `blog/src/content/posts/` 下新建 `.md` 文件：

```markdown
---
author: Link
pubDatetime: 2026-08-11T22:00:00.000Z
title: 文章标题
featured: false
draft: false
tags:
  - 标签1
  - 标签2
description: 文章摘要
---

正文内容...
```

### 写笔记

在 `note/src/content/docs/` 对应分类目录下新建 `.mdx` 文件。

例如 `note/src/content/docs/deep-learning/transformer.mdx`：

```markdown
---
title: Transformer 详解
description: Transformer 架构的核心原理
---

## 注意力机制
...
```

笔记站会根据文件目录结构自动生成左侧导航栏。

## 部署

推送到 `main` 分支后，Vercel 自动部署：

```bash
git add .
git commit -m "新文章：xxx"
git push
```

- **blog**：Vercel Root Directory 设为 `blog`
- **note**：Vercel Root Directory 设为 `note`

## 许可证

博客主题基于 [AstroPaper](https://github.com/satnaing/astro-paper) (MIT)，笔记站基于 [Starlight](https://github.com/withastro/starlight) (MIT)。

本站内容采用 MIT 许可证。
