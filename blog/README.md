# 博客站

`blog/` 是基于 AstroPaper 的个人博客，发布技术文章、项目记录和独立页面，部署地址为 <https://link-blog.vercel.app>。

## 常用命令

```bash
npm ci
npm run dev
npm run build
npm run preview
npm run lint
npm run format:check
```

开发服务器默认运行在 `http://localhost:4321`。

## 内容

- `src/content/posts/`：文章（`.md` / `.mdx`）
- `src/content/pages/`：独立页面，例如关于页
- `src/content.config.ts`：文章 front matter 的类型约束

文章列表、标签、归档、搜索、RSS 和 Sitemap 都由 `src/pages/` 下的路由生成。站点信息和功能开关集中在 `astro-paper.config.ts`。

## 构建流程

`npm run build` 会依次执行类型检查、Astro 静态构建和 Pagefind 搜索索引生成。
