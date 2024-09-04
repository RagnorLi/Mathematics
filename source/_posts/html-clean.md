---
title: HTML-Clean
date: 2024-09-03 14:36:42
background: bg-[#cc5534]
tags:
  - web
categories:
  - Frontend
intro: |
  HTML 就是网页的骨架
plugins:
  - copyCode
---


## Hypertext Markup Language


### 实体 

```html
&nbsp;  <!-- 空格 -->
&gt;    <!-- > -->
&lt;    <!-- < -->
&amp;   <!-- & -->
&copy; <!-- © -->
```

### `<meta>` 

```html
<meta charset="UTF-8"> <!-- 字符集 -->
<meta name="keywords" content=""> <!-- 搜索引擎关键字 -->
<meta name="description" content=""> <!-- 网站描述 -->
<meta http-equiv="refresh" content="3;url=___"> <!-- 重定向 -->
```

### `<title>` 

```html
<title>标题</title>
```

### 语义化标签 

```html
<h1>标题1</h1>
<p>段落内容</p>
<em>斜体强调</em>
<strong>加粗强调</strong>
<q>短引用</q>
<blockquote>长引用</blockquote>
<hr> <!-- 分割线 -->
<br> <!-- 换行 -->
```

### 布局标签 

```html
<header>网页头部</header>
<main>网页主体</main>
<footer>网页底部</footer>
<nav>网页导航</nav>
<aside>侧边内容</aside>
<article>文章标题</article>
<section>独立区块</section>
<div>div块</div>
<span>span行内元素</span>
```

### 列表 

```html
<ul><li>无序列表项</li></ul>
<ol><li>有序列表项</li></ol>
<dl><dt>名词</dt><dd>描述</dd></dl>
```

### 超链接 

```html
<a href="">超链接</a>
<a href="./">相对路径 (本文件所在目录)</a>
<a href="../">相对路径 (上一级目录)</a>
<a href="#" target="black">新页面打开</a>
<a href="#" target="self">当前页面打开</a>
<a href="#">回顶部</a>
<a href="#元素id">去指定位置</a>
<a href="javascript:;">无意义链接</a>
```

### 图片标签

```html
<img src="" alt="描述" width="100%">
```

### 媒体标签 

#### 视频标签
```html
<video controls width="100%">
  <source src="https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4" type="video/mp4">
  Sorry, your browser doesn't support embedded videos.
</video>
```

### 内联框架 

#### iFrame 示例
```html
<iframe
  title="示例框架"
  width="342"
  height="306"
  src="https://example.com"
  scrolling="no"
></iframe>
```

### 表格 

```html
<table>
  <tr>
    <td>单元格1</td>
    <td>单元格2</td>
  </tr>
  <tr>
    <td rowspan="2">合并行</td>
    <td colspan="2">合并列</td>
  </tr>
</table>
```

### 表单 

#### 表单结构

```html
<form action="*.html">
  <input type="text">
  <input type="password">
  <input type="color">
  <input type="email">
  <input type="submit">
  <input type="reset">
  <input type="radio" value="" checked>
  <input type="checkbox" value="" checked>
</form>
```

#### 下拉列表

```html
<form action="*.html">
  <select name="haha">
    <option value="i">A</option>
    <option value="ii" selected>B</option>
    <option value="iii">C</option>
  </select>
</form>






