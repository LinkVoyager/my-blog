# Link 的个人站点

这是一个由两个独立 Astro 站点组成的个人内容仓库：博客用于发布成文文章，笔记站用于维护按主题组织的知识库。

## 站点

| 目录 | 用途 | 地址 | 技术栈 |
|---|---|---|---|
| `blog/` | 技术文章、项目记录 | [link-blog.vercel.app](https://link-blog.vercel.app) | Astro + AstroPaper + Tailwind |
| `note/` | 学习笔记、知识库 | [links-note.vercel.app](https://links-note.vercel.app) | Astro + Starlight |

两个目录都是独立项目，各自拥有 `package.json` 和 `package-lock.json`，统一使用 npm。

## 目录结构

```text
E:/Blog/
├── blog/                    # 博客站
│   ├── src/content/posts/   # 博客文章（.md / .mdx）
│   ├── src/content/pages/   # 独立页面
│   ├── src/pages/           # Astro 路由
│   ├── src/components/      # 公共组件
│   ├── src/layouts/         # 页面布局
│   ├── src/utils/           # 文章处理和 URL 工具
│   ├── astro.config.ts       # Astro、Markdown、i18n 配置
│   └── astro-paper.config.ts # 站点信息和功能开关
├── note/                    # 笔记站
│   ├── src/content/docs/    # 笔记内容（.md / .mdx）
│   ├── src/styles/          # 自定义样式
│   └── astro.config.mjs     # Starlight 侧边栏和站点配置
└── .github/workflows/ci.yml # 两个站点的持续集成
```

## 本地开发

在两个终端中分别启动：

```bash
cd blog
npm ci
npm run dev              # http://localhost:4321
```

```bash
cd note
npm ci
npm run dev              # http://localhost:4322
```

## 写内容

博客文章放在 `blog/src/content/posts/`，需要填写 `title`、`description`、`pubDatetime` 和 `tags` 等 front matter。完整字段约束见 `blog/src/content.config.ts`。

笔记放在 `note/src/content/docs/` 对应分类目录中。侧边栏会根据目录自动生成，新增 Markdown 文件后即可出现对应导航项。

## 构建

```bash
cd blog && npm run build
cd note && npm run build
```

博客构建还会生成 Pagefind 搜索索引；两个站点的 `dist/` 都是构建产物，不是源码。

## 部署

两个站点分别连接到 Vercel：

- 博客的 Root Directory：`blog`
- 笔记站的 Root Directory：`note`

推送到 `main` 后由 Vercel 自动构建和部署。
