// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
  site: 'https://note.example.com', // 部署后替换为实际域名
  integrations: [
    starlight({
      title: 'Link 的笔记',
      defaultLocale: 'zh-CN',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/LinkVoyager',
        },
      ],
      sidebar: [
        {
          label: '深度学习',
          items: [{ autogenerate: { directory: 'deep-learning' } }],
        },
        {
          label: '代码算法',
          items: [{ autogenerate: { directory: 'code-algorithm' } }],
        },
        {
          label: '工具使用',
          items: [{ autogenerate: { directory: 'tools' } }],
        },
        {
          label: '随想记录',
          items: [{ autogenerate: { directory: 'minds' } }],
        },
      ],
    }),
  ],
});
