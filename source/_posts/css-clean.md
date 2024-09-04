---
title: CSS-Clean
date: 2024-09-03 14:36:42
background: bg-[#cc5534]
tags:
  - web
categories:
  - Frontend
intro: |
  CSS 就是网页的化妆师
plugins:
  - copyCode
---

## Cascading Style Sheets


### CSS简介

修改元素样式的三种方式：
1. **内联样式**（行内样式）
2. **内部样式表**
3. **外部样式表**

### CSS语法

```css
选择器 {
  声明块（名值对）
}
```

### 选择器

#### 标签选择器

```css
标签名 {
  /* 元素选择器 */
}
```

#### ID选择器

```css
#id {
  /* ID选择器 */
}
```

#### 类选择器

```css
.class {
  /* 类选择器 */
}
```

#### 通配选择器

```css
* {
  /* 通配选择器 */
}
```

#### 交集选择器

```css
a.b {
  /* 交集选择器 */
}
```

#### 并集选择器

```css
a, b, c {
  /* 并集选择器 */
}
```

#### 子元素选择器

```css
parent > son {
  /* 子元素选择器 */
}
```

#### 后代元素选择器

```css
parent son {
  /* 后代元素选择器 */
}
```

#### 毗邻兄弟元素选择器

```css
son + sibling {
  /* 毗邻兄弟元素选择器（下一个） */
}
```

#### 通用兄弟元素选择器

```css
son ~ sibling {
  /* 通用兄弟元素选择器（其后所有） */
}
```

#### 属性选择器

```css
a[自定义属性名] {
  /* 属性选择器 */
}
```

#### 指定属性值选择器

```css
a[自定义属性名=""] {
  /* 指定属性值选择器 */
}
```

#### 指定字符开头选择器

```css
a[自定义属性名^=""] {
  /* 指定字符开头选择器 */
}
```

#### 指定字符结束选择器

```css
a[自定义属性名$=""] {
  /* 指定字符结束选择器 */
}
```

#### 包含指定字符选择器

```css
a[自定义属性名*=""] {
  /* 包含指定字符选择器 */
}
```

#### 包含指定字符串选择器

```css
a[自定义属性名~=""] {
  /* 包含指定字符串选择器（与其它字符串用空格分隔开的） */
}
```

### 伪类

#### :first-child

```css
li:first-child {
  /* 其兄弟元素中的第一个 */
}
```

#### :last-child

```css
li:last-child {
  /* 其兄弟元素中的最后一个 */
}
```

#### :nth-child(n)

```css
li:nth-child(n) {
  /* 其兄弟元素中的某个 */
}
```

#### :first-of-type

```css
li:first-of-type {
  /* 其兄弟元素中同类型的第一个 */
}
```

#### :last-of-type

```css
li:last-of-type {
  /* 其兄弟元素中同类型的最后一个 */
}
```

#### :nth-of-type(n)

```css
li:nth-of-type(n) {
  /* 其兄弟元素中同类型的某个 */
}
```

#### :nth-last-of-type(n)

```css
li:nth-last-of-type(n) {
  /* 其兄弟元素中同类型的倒数某个 */
}
```

#### :not(*)

```css
li:not(*) {
  /* 除了*的全部 */
}
```

#### :visited

```css
a:visited {
  /* 访问过的链接 */
}
```

#### :link

```css
a:link {
  /* 未访问的链接 */
}
```

#### :hover

```css
a:hover {
  /* 鼠标移入状态 */
}
```

#### :active

```css
a:active {
  /* 鼠标按下状态 */
}
```

### 伪元素

#### ::first-letter

```css
::first-letter {
  /* 第一个字母 */
}
```

#### ::first-line

```css
::first-line {
  /* 第一行 */
}
```

#### ::selection

```css
::selection {
  /* 被鼠标长按选中的 */
}
```

#### ::before

```css
::before {
  /* 元素的开始（配合content属性） */
}
```

#### ::after

```css
::after {
  /* 元素的末尾（配合content属性） */
}
```

### 选择器权重

#### !important

```css
property: value !important;
```
- **功能**: 最高优先级。

#### 内联样式

```html
<div style="color: red;">
```
- **优先级**: 1，0，0，0。

#### ID选择器优先级

```css
#id {
  /* ID选择器的优先级高于类选择器 */
}
```
- **优先级**: 0，1，0，0。

#### 类和伪类选择器优先级

```css
.class {
  /* 类和伪类选择器 */
}
```
- **优先级**: 0，0，1，0。

#### 元素选择器优先级

```css
tag {
  /* 元素选择器优先级最低 */
}
```
- **优先级**: 0，0，0，1。

#### 通配选择器优先级

```css
* {
  /* 通配选择器 */
}
```
- **优先级**: 0，0，0，0。

#### 继承样式

- **优先级**: 没有优先级。

### CSS原生变量

```css
:root {
  --变量名: 值;
}

元素选择器 {
  width: var(--变量名);
}
```
- **示例**:
```css
html {
  --length: 200px;
}

.box1 {
  width: var(--length);
}
```

### Less 特点

1. 数值可以直接进行计算。
2. 可以用 `@import` 把其它 less 引入。
3. 可以用 `@a: 100px;` 来存储变量。
4. 父元素的样式包裹子元素的样式。
5. `p1:extend(.p2) {}` p1 扩展 p2 的样式。
6. 混合函数，设置变量，下次直接用。

### 显示/隐藏

#### display

```css
display: block; /* 显示 */
display: none;  /* 不显示，不占位 */
```

#### visibility

```css
visibility: visible; /* 显示，占位 */
visibility: hidden;  /* 不显示，占位 */
```

### 鼠标状态

```css
pointer-events: none; /* 鼠标事件探测禁用 */
```

### 盒子模型

#### 盒子水平布局

```css
1. margin-left;
2. border-left;
3. padding-left;
4. width;
5. padding-right;
6. border-right;
7. margin-right;
```

#### 溢出的处理方式

```css
overflow: visible; /* 显示溢出内容 */
overflow: hidden;  /* 隐藏溢出内容，脱离文档流 */
overflow: scroll;  /* 添加滚动条 */
overflow: auto;    /* 自动添加滚动条 */
```

#### 盒子大小

```css
box-sizing: content-box; /* 盒子大小不包括边框 */
box-sizing: border-box;  /* 盒子大小包括边框 */
```

#### 轮廓线

```css
outline: 1px solid black;
```

#### 阴影

```css
box-shadow: 10px 10px 5px #888888;
```

#### 圆角

```css
border-radius: 10px;
```

### BFC

#### 开启BFC

```css
float: left;
display: inline-block;
overflow: hidden; /* 非 visible */
```
- **BFC（Block Formatting Context）块级格式化**: CSS中的一个隐含的属性，可以为一个元素开启BFC。

#### BFC的作用

1. 开启BFC的元素不会被浮动元素所覆盖。
2. 开启BFC的元素子元素和父元素的外边距不会重叠。
3. 开启BFC可以包含浮动的子元素。

### 浮动

#### 清除浮动元素的影响

```css
clear: left;
clear: right;
clear: both;
```

### Clearfix

```css
.clearfix::before,
.clearfix::after {
  content: "";
  display: table;
  clear: both;
}
```
- **功能**: 解决高度塌陷的最终方案。

### 定位

#### position: static

```css
position: static; /* 默认值，静态定位 */
```

#### position: relative

```css
position: relative; /* 相对定位，不脱离文档流 */
``


#### position: absolute

```css
position: absolute;
```
- **功能**: 绝对定位，脱离文档流。位置是相对于最近的已定位的祖先元素（即包含块）进行定位。如果没有已定位的祖先元素，则相对于视口（根元素）进行定位。`inline`元素会变成`block`元素。

#### position: fixed

```css
position: fixed;
```
- **功能**: 固定定位，脱离文档流。与绝对定位类似，但固定在视口的某个位置，滚动页面时元素不会移动。

#### position: sticky

```css
position: sticky;
```
- **功能**: 粘滞定位，不脱离文档流。当页面滚动到特定位置时，元素会固定在该位置。注意：`IE`浏览器不兼容。

### 文本样式

#### font-family

```css
font-family: "serif", "sans-serif", "monospace";
```
- **功能**: 设置文本的字体。`serif`表示衬线字体，`sans-serif`表示非衬线字体，`monospace`表示等宽字体。

#### text-align

```css
text-align: left;
text-align: right;
text-align: center;
text-align: justify;
```
- **功能**: 设置文本的水平对齐方式。`left`左对齐，`right`右对齐，`center`居中对齐，`justify`两端对齐。

#### vertical-align

```css
vertical-align: baseline;
vertical-align: top;
vertical-align: bottom;
vertical-align: middle;
```
- **功能**: 设置元素的垂直对齐方式。常用于`img`图片和`inline`元素对齐。`baseline`是默认值，表示基线对齐，`top`顶部对齐，`bottom`底部对齐，`middle`居中对齐。还可以直接指定具体的像素值进行对齐。

#### direction

```css
direction: rtl;
direction: ltr;
```
- **功能**: 设置文字的方向。`rtl`表示从右到左，`ltr`表示从左到右。

#### text-overflow: ellipsis

```css
text-overflow: ellipsis;
```
- **功能**: 当文本溢出时显示省略号。必须与以下两个属性一起使用：
```css
overflow: hidden;
white-space: nowrap;
```

#### white-space

```css
white-space: normal;
white-space: nowrap;
white-space: pre;
```
- **功能**: 设置元素如何处理空白区域。`normal`表示正常换行，`nowrap`表示不换行且去掉空白，`pre`表示保留空白。

### 背景样式

#### background-image

```css
background-image: url('image.png');
```
- **功能**: 设置元素的背景图片。

#### background-color

```css
background-color: linear-gradient(to right, red, blue);
background-color: radial-gradient(circle at center, red, blue);
```
- **功能**: 设置元素的背景颜色或渐变背景。支持`linear-gradient`线性渐变和`radial-gradient`径向渐变。

#### background-repeat

```css
background-repeat: repeat;
background-repeat: no-repeat;
background-repeat: repeat-x;
background-repeat: repeat-y;
```
- **功能**: 控制背景图片的重复方式。

#### background-position

```css
background-position: left top;
background-position: center center;
background-position: right bottom;
```
- **功能**: 设置背景图片的位置。

#### background-clip

```css
background-clip: border-box;
background-clip: padding-box;
background-clip: content-box;
```
- **功能**: 设置背景的绘制区域。`border-box`表示包括边框，`padding-box`表示不包括边框但包括内边距，`content-box`表示只绘制在内容区域。

#### background-origin

```css
background-origin: border-box;
background-origin: padding-box;
background-origin: content-box;
```
- **功能**: 设置背景图片的定位区域。

#### background-size

```css
background-size: 100px 100px;
background-size: cover;
background-size: contain;
```
- **功能**: 设置背景图片的大小。`cover`表示覆盖整个容器，`contain`表示背景图片完全显示在容器内且保持比例。

#### background-attachment

```css
background-attachment: scroll;
background-attachment: fixed;
```
- **功能**: 设置背景图片是否随页面滚动。`scroll`表示背景随元素移动，`fixed`表示背景固定。

### 过渡效果

#### transition-property

```css
transition-property: all;
```
- **功能**: 指定要执行过渡的属性。可以使用`all`关键字表示所有属性，或者指定具体的属性名。

#### transition-duration

```css
transition-duration: 1s;
```
- **功能**: 指定过渡效果的持续时间。时间单位为`秒(s)`或`毫秒(ms)`。

#### transition-timing-function

```css
transition-timing-function: ease;
transition-timing-function: linear;
transition-timing-function: ease-in;
transition-timing-function: ease-out;
transition-timing-function: ease-in-out;
```
- **功能**: 指定过渡的时序函数，控制过渡的执行方式。`ease`为默认值，`linear`为匀速，`ease-in`为加速，`ease-out`为减速，`ease-in-out`为先加速后减速。还可以使用贝塞尔曲线`cubic-bezier()`自定义。

#### transition-delay

```css
transition-delay: 0.5s;
```
- **功能**: 指定过渡效果的延迟时间。

### 动画效果

#### @keyframes

```css
@keyframes name {
  from { /* 初始状态 */ }
  to { /* 结束状态 */ }
}
```
- **功能**: 定义动画的关键帧。

#### animation-name

```css
animation-name: name;
```
- **功能**: 设置要对当前元素生效的关键帧的名字。

#### animation-duration

```css
animation-duration: 2s;
```
- **功能**: 设置动画的执行时间。

#### animation-delay

```css
animation-delay: 0.5s;
```
- **功能**: 设置动画的延迟时间。

#### animation-iteration-count

```css
animation-iteration-count: infinite;
animation-iteration-count: 3;
```
- **功能**: 设置动画的执行次数。可以设置为具体的次数或`infinite`表示无限循环。

#### animation-direction

```css
animation-direction: normal;
animation-direction: reverse;
animation-direction: alternate;
animation-direction: alternate-reverse;
```
- **功能**: 设置动画的执行方向。`normal`表示从`from`到`to`，`reverse`表示从`to`到`from`，`alternate`表示正反交替执行，`alternate-reverse`表示反正交替执行。

#### animation-play-state

```css
animation-play-state: running;
animation-play-state: paused;
```
- **功能**: 设置动画的执行状态。`running`为默认值，表示动画正在执行，`paused`表示动画暂停。

#### animation-fill-mode

```css
animation-fill-mode: none;
animation-fill-mode: forwards;
animation-fill-mode: backwards;
animation-fill-mode: both;
```
- **功能**: 设置动画的填充模式。`none`表示动画执行完毕元素回到原来位置，`forwards`表示动画结束后元素保持在结束位置，`backwards`表示动画延时等待时，元素处于开始位置，`both`结合`forwards`和`backwards`的效果。

### Transform 形变

#### translateX() translateY()

```css
transform: translateX(100px);
transform: translateY(50px);
```
- **功能**: 平移元素。`Z`轴的平移需要设置`perspective`属性。

#### rotateX() rotateY() rotateZ()

```css
transform: rotateX(45deg);
transform: rotateY(45deg);
transform: rotateZ(45deg);
```
- **功能**: 旋转元素。

#### scaleX() scaleY() scale()

```css
transform: scaleX(1.5);
transform: scaleY(2);
transform: scale(1.5, 2);
```
- **功能**: 缩放元素。`scale()`可以同时缩放`X`和`Y`方向。

#### transform-origin

```css
transform-origin: 50% 50%;
```
- **功能**: 设置变形的原点位置。

#### transform-style: preserve-3d

```css
transform-style: preserve-3d;
```
- **功能**: 设给父元素，保留子元素的`3D`效果。

### 弹性容器

#### flex-direction

```css
flex-direction: row;
flex-direction: row-reverse;
flex-direction: column;
flex-direction: column-reverse;
```
- **功能**: 设置弹性容器中元素在主轴（主方向）上的排列方式。`row`表示水平排列，`row-reverse`表示反向水平排列，`column`表示纵向排列，`column-reverse`表示反向纵向排列。

#### flex-wrap

```css
flex-wrap: nowrap;
flex-wrap: wrap;
flex-wrap: wrap-reverse;
```
- **功能**: 设置弹性元素是否自动换行。`nowrap`表示不换行，`wrap`表示自动换行，`wrap-reverse`表示反向换行。

#### flex-flow

```css
flex-flow: column wrap;
```
- **功能**: `flex-direction`和`flex-wrap`的简写属性。可以同时设置主轴方向和换行方式。

#### justify-content

```css
justify-content: flex-start;
justify-content: flex-end;
justify-content: center;
justify-content: space-between;
justify-content: space-around;
justify-content: space-evenly;
```
- **功能**: 如何分配弹性容器内的空白空间，即如何排列主轴上的元素。`flex-start`表示从主轴的起始边排列，`flex-end`表示从主轴的终边排列，`center`表示居中排列，`space-between`表示元素间隔等距排列且两端没有空白，`space-around`表示元素间隔等距排列且两端有空白，`space-evenly`表示所有间距和两端空白相等（`IE`不兼容）。

#### align-items

```css
align-items: stretch;
align-items: flex-start;
align-items: flex-end;
align-items: center;
align-items: baseline;
```
- **功能**: 设置弹性容器内的元素在辅轴（辅方向）上的排列方式。`stretch`表示拉伸元素以填满容器，`flex-start`表示从辅轴的起始边排列，`flex-end`表示从辅轴的终边排列，`center`表示居中排列，`baseline`表示基线对齐。

#### align-content

```css
align-content: flex-start;
align-content: flex-end;
align-content: center;
align-content: space-between;
align-content: space-around;
align-content: stretch;
```
- **功能**: 设置弹性容器在辅轴上的空白空间的分布方式。`flex-start`表示从辅轴的起始边排列，`flex-end`表示从辅轴的终边排列，`center`表示居中排列，`space-between`表示空白间隔等距排列且两端没有空白，`space-around`表示空白间隔等距排列且两端有空白，`stretch`表示拉伸元素以填满空白。

### 弹性元素

#### align-self

```css
align-self: flex-start;
```
- **功能**: 覆盖当前弹性元素上的对齐方式，为某个弹性元素单独设置对齐方式。

#### flex-grow

```css
flex-grow: 1;
```
- **功能**: 设置弹性元素的伸展系数。数值越大，元素越容易伸展占据多余的空间。

#### flex-shrink

```css
flex-shrink: 1;
```
- **功能**: 设置弹性元素的收缩系数。数值越大，元素越容易收缩以适应容器的大小。

#### flex-basis

```css
flex-basis: auto;
flex-basis: 100px;
```
- **功能**: 设置元素在主轴上的基础长度。如果主轴是横向的，则该值指定元素的宽度；如果主轴是纵向的，则该值指定元素的高度。默认值为`auto`，表示参考元素自身的高度/宽度。

#### flex

```css
flex: 1;
flex: 0 1 auto;
flex: none;
```
- **功能**: `flex-grow`、`flex-shrink`和`flex-basis`的简写属性。`initial`等于`flex: 0 1 auto`，`auto`等于`flex: 1 1 auto`，`none`等于`flex: 0 0 auto`。

#### order

```css
order: 1;
order: 2;
order: 3;
```
- **功能**: 用于决定元素的排列顺序。数值越小，元素排列越靠前。






