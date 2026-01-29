#!/bin/bash

echo "🧹 开始清理 Git 仓库中的垃圾文件..."

# 1. 创建或覆盖 .gitignore 文件
# 这里定义了所有需要忽略的文件类型
cat > .gitignore << EOF
# === Python 自动生成 ===
__pycache__/
*.py[cod]
*$py.class

# === 编译与打包 ===
*.egg-info/
.eggs/
dist/
build/
develop-eggs/
lib/
lib64/

# === 训练日志与产物 (最占空间的部分) ===
logs/
outputs/
videos/
runs/
*.tfevents*

# === 模型权重文件 ===
*.pt
*.pth
*.ckpt
*.onnx
*.safetensors

# === IDE 配置 ===
.vscode/
.idea/
*.swp

# === 其他 ===
.DS_Store
EOF

echo "✅ .gitignore 文件已更新。"

# 2. 从 Git 缓存中移除这些文件 (不删本地文件)
# 这步是为了解决“已经 commit 过的文件不受 gitignore 限制”的问题
echo "🔄 正在从 Git 索引中移除忽略的文件 (请稍候)..."

git rm -r --cached logs/ 2>/dev/null
git rm -r --cached outputs/ 2>/dev/null
git rm -r --cached *.egg-info/ 2>/dev/null
git rm -r --cached **/context.pkl 2>/dev/null
git rm -r --cached **/__pycache__/ 2>/dev/null
git rm -r --cached **/*.pyc 2>/dev/null
git rm -r --cached **/*.pt 2>/dev/null

echo "✅ 垃圾文件已从 Git 追踪列表中移除。"

# 3. 重新添加剩下的文件
echo "➕ 重新添加有效文件..."
git add .

echo "-------------------------------------------------------"
echo "🎉 清理完成！"
echo "👉 现在请运行以下命令提交更改："
echo "   git commit -m 'chore: update gitignore and remove large files'"
echo "   git push"
echo "-------------------------------------------------------"
