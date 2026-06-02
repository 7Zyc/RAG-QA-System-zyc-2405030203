# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('docs', 'docs'),
    ],
    hiddenimports=[
        'langchain_community.llms.ollama',
        'langchain_community.embeddings.ollama',
        'langchain_community.vectorstores.chroma',
        'langchain_community.document_loaders.pdf',
        'langchain_community.document_loaders.word_document',
        'langchain_community.document_loaders.text',
        'langchain_text_splitters',
        'chromadb',
        'streamlit',
        'tiktoken',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'PIL',
        'notebook',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RAG-QA-System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RAG-QA-System',
)
