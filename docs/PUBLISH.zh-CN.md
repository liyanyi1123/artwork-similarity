# 发布到 GitHub

应上传 `artwork-similarity-github/` 内的内容。本地整个研究目录包含数据、权重和已有 Git 历史，不作为这次发布的仓库根目录。

本次已准备 README、研究代码、结果证据、运行入口、依赖与忽略规则。项目作者尚未指定代码许可证；发布时可按自己的意愿补充 `LICENSE`。现有第三方声明见 `THIRD_PARTY_NOTICES.md`。

如果使用 Git 命令行，在这个新目录内执行：

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Organize artwork similarity research and reproducible results"
```

在 GitHub 建立空仓库后，将下面的 `YOUR_USERNAME` 和仓库名替换成实际值：

```bash
git remote add origin https://github.com/YOUR_USERNAME/artwork-similarity.git
git push -u origin main
```

以上命令用于首次发布或另建副本；已有 `.git` 的工作副本无需重复初始化。压缩包用于下载、备份和解压；GitHub 仓库展示解压后的项目内容。
