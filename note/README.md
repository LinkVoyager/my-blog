# 笔记站

`note/` 是基于 Astro Starlight 的中文知识库，部署地址为 <https://links-note.vercel.app>。

## 常用命令

```bash
npm ci
npm run dev
npm run build
npm run preview
```

开发服务器固定运行在 `http://localhost:4322`，可与博客站同时启动。

## 内容

所有笔记放在 `src/content/docs/`：

```text
src/content/docs/
├── deep-learning/   # 深度学习
├── code-algorithm/  # 代码算法
├── tools/           # 工具使用
└── minds/           # 随想记录
```

每个目录的侧边栏项目由 `astro.config.mjs` 的 `autogenerate` 自动生成。笔记集合使用 `src/content.config.ts` 中的 Starlight `docsLoader` 和 `docsSchema`。

Markdown 已启用数学公式支持，样式位于 `src/styles/math.css`。
