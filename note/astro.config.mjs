// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeMathjax from "rehype-mathjax";

// https://astro.build/config
export default defineConfig({
  site: "https://links-note.vercel.app",
  server: { port: 4322 },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeMathjax],
    }),
  },
  integrations: [
    starlight({
      title: "Link 的笔记",
      defaultLocale: "zh-CN",
      customCss: ["./src/styles/math.css"],
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/LinkVoyager",
        },
      ],
      sidebar: [
        {
          label: "深度学习",
          items: [{ autogenerate: { directory: "deep-learning" } }],
        },
        {
          label: "代码算法",
          items: [{ autogenerate: { directory: "code-algorithm" } }],
        },
        {
          label: "工具使用",
          items: [{ autogenerate: { directory: "tools" } }],
        },
        {
          label: "随想记录",
          items: [{ autogenerate: { directory: "minds" } }],
        },
      ],
    }),
  ],
});
