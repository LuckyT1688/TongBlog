/**
 * 全局配置文件
 * 统一管理项目中的常量和配置项
 */

// 分页配置
export const PAGINATION_CONFIG = {
  // 博客文章每页显示数量
  BLOG_ENTRIES_PER_PAGE: 10,
  // 默认分页大小（与 astro.config.mjs 保持一致）
  DEFAULT_PAGE_SIZE: 8,
} as const;

// 从 astro.config.mjs 获取配置（如果需要的话）
// 注意：在 Astro 中，config 文件的配置通常不能直接在运行时访问
// 这里提供一个统一的配置管理方式


// 导出便捷的获取函数
export const getPageSize = (type: 'blog' = 'blog'): number => {
  switch (type) {
    case 'blog':
      return PAGINATION_CONFIG.BLOG_ENTRIES_PER_PAGE;
    default:
      return PAGINATION_CONFIG.DEFAULT_PAGE_SIZE;
  }
};

// 类型定义
export type PageType = 'blog';

//网站信息
export const SITE_INFO = {
  // 网站名称
  NAME: 'Tong的博客',
  SITE_NAME: 'Tong',
  SUBNAME: '数据分析师的技术分享空间',
  // 网站描述
  DESCRIPTION: 'Tong的个人博客，专注于数据分析、Python数据处理等技术分享。',
  // 网站 URL (生产环境)
  URL: 'https://tongblog-61e.pages.dev',
  AUTHOR: 'Tong',
  // 本地开发 URL
  DEV_URL: 'http://localhost:4321',
  LOGO_IMAGE: '/favicon/立里.jpg',
  KEY_WORDS: '数据分析,Python,Tong',
  GOOGLE_ANALYTICS_ID: 'G-XXXXXX',  // 需改为你自己的Google Analytics ID
  BAIDU_ANALYTICS_ID: 'XXXXXXXXXX', // 需改为你自己的百度分析ID
  // GitHub 仓库
  GITHUB_REPO: 'https://github.com/LuckyT1688/TongBlog',
  // 联系邮箱
  EMAIL: 'contact@litong.asia',
  // 网站初始时间（用于计算运行时长）
  START_DATE: '2025-11-14',
  // ICP 备案信息
  ICP: {
    NUMBER: '',
    URL: ''
  }
} as const;

// 获取当前环境的网站URL
export const getSiteUrl = () => {
  // 在构建时使用生产URL，开发时使用开发URL
  return import.meta.env.PUBLIC_ENV === 'production' ? SITE_INFO.URL : SITE_INFO.DEV_URL;
};
