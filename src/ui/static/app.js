// Novel RAG Chatbot - Frontend JavaScript

class NovelRAG {
    constructor() {
        this.activeNovel = null;
        this.novels = [];
        this.pendingFile = null;
        this.queryInFlight = false;

        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.loadTheme();
        this.loadNovels();
    }

    bindElements() {
        // Header
        this.libraryToggle = document.getElementById('libraryToggle');
        this.themeToggle = document.getElementById('themeToggle');
        this.activeNovelDisplay = document.getElementById('activeNovel');

        // Library panel
        this.libraryPanel = document.getElementById('libraryPanel');
        this.closeLibrary = document.getElementById('closeLibrary');
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.novelsList = document.getElementById('novelsList');

        // Chat
        this.messages = document.getElementById('messages');
        this.welcomeMessage = document.getElementById('welcomeMessage');
        this.openLibraryBtn = document.getElementById('openLibraryBtn');
        this.queryForm = document.getElementById('queryForm');
        this.queryInput = document.getElementById('queryInput');
        this.submitBtn = document.getElementById('submitBtn');

        // Sources
        this.sourcesPanel = document.getElementById('sourcesPanel');
        this.closeSources = document.getElementById('closeSources');
        this.sourcesList = document.getElementById('sourcesList');

        // Modals
        this.processingModal = document.getElementById('processingModal');
        this.processingTitle = document.getElementById('processingTitle');
        this.processingStatus = document.getElementById('processingStatus');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');

        this.uploadModal = document.getElementById('uploadModal');
        this.uploadForm = document.getElementById('uploadForm');
        this.bookTitle = document.getElementById('bookTitle');
        this.bookAuthor = document.getElementById('bookAuthor');
        this.cancelUpload = document.getElementById('cancelUpload');
    }

    bindEvents() {
        // Library toggle
        this.libraryToggle.addEventListener('click', () => this.toggleLibrary());
        this.closeLibrary.addEventListener('click', () => this.toggleLibrary(false));
        this.openLibraryBtn?.addEventListener('click', () => this.toggleLibrary(true));

        // Theme toggle
        this.themeToggle.addEventListener('click', () => this.toggleTheme());

        // Upload
        this.uploadZone.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('dragover');
        });
        this.uploadZone.addEventListener('dragleave', () => {
            this.uploadZone.classList.remove('dragover');
        });
        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                this.handleFile(e.dataTransfer.files[0]);
            }
        });

        // Upload form
        this.uploadForm.addEventListener('submit', (e) => this.handleUploadSubmit(e));
        this.cancelUpload.addEventListener('click', () => this.closeUploadModal());

        // Query form
        this.queryForm.addEventListener('submit', (e) => this.handleQuery(e));
        this.queryInput.addEventListener('input', () => this.autoResizeTextarea());
        this.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.queryForm.dispatchEvent(new Event('submit'));
            }
        });

        // Sources
        this.closeSources?.addEventListener('click', () => this.toggleSources(false));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'k' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.toggleLibrary();
            }

            if (e.key === '/' && document.activeElement !== this.queryInput) {
                e.preventDefault();
                if (!this.queryInput.disabled) {
                    this.queryInput.focus();
                }
            }
        });
    }

    // === Theme ===

    loadTheme() {
        const theme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', theme);
    }

    toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
    }

    // === Library ===

    toggleLibrary(open = null) {
        if (open === null) {
            this.libraryPanel.classList.toggle('open');
        } else if (open) {
            this.libraryPanel.classList.add('open');
        } else {
            this.libraryPanel.classList.remove('open');
        }
    }

    async loadNovels() {
        try {
            const response = await fetch('/api/novels');
            const data = await response.json();
            this.novels = data.novels || [];
            this.renderNovelsList();
            await this.loadActiveNovel();
        } catch (error) {
            console.error('Failed to load novels:', error);
        }
    }

    async loadActiveNovel() {
        try {
            const response = await fetch('/api/novels/active');
            const data = await response.json();
            if (data.active) {
                this.setActiveNovel(data.active);
            }
        } catch (error) {
            console.error('Failed to load active novel:', error);
        }
    }

    renderNovelsList() {
        if (!this.novels.length) {
            this.novelsList.innerHTML = '<p class="empty-state">No books yet. Upload one to get started!</p>';
            return;
        }

        this.novelsList.innerHTML = this.novels.map(novel => `
            <div class="novel-card ${novel.id === this.activeNovel?.id ? 'active' : ''}" data-id="${novel.id}">
                <div class="novel-icon ${novel.format}">
                    ${this.getFormatIcon(novel.format)}
                </div>
                <div class="novel-info">
                    <div class="novel-title">${this.escapeHtml(novel.title)}</div>
                    <div class="novel-meta">${novel.chapters_indexed} chapters · ${novel.chunks_count} chunks</div>
                </div>
                <div class="novel-actions">
                    <button class="novel-action delete" title="Delete" data-id="${novel.id}">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18"></path>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"></path>
                            <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
            </div>
        `).join('');

        // Bind events
        this.novelsList.querySelectorAll('.novel-card').forEach(card => {
            card.addEventListener('click', (e) => {
                if (!e.target.closest('.novel-actions')) {
                    this.selectNovel(card.dataset.id);
                }
            });
        });

        this.novelsList.querySelectorAll('.novel-action.delete').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteNovel(btn.dataset.id);
            });
        });
    }

    getFormatIcon(format) {
        const icons = {
            txt: '📄',
            pdf: '📕',
            epub: '📗'
        };
        return icons[format] || '📖';
    }

    async selectNovel(novelId) {
        try {
            const response = await fetch(`/api/novels/${novelId}/select`, {
                method: 'POST'
            });

            if (!response.ok) {
                const error = await response.json();
                alert(error.detail || 'Failed to select novel');
                return;
            }

            const data = await response.json();
            this.setActiveNovel(data.novel);
            this.toggleLibrary(false);

        } catch (error) {
            console.error('Failed to select novel:', error);
            alert('Failed to select novel');
        }
    }

    setActiveNovel(novel) {
        this.activeNovel = novel;

        // Update UI
        if (novel) {
            this.activeNovelDisplay.classList.add('has-novel');
            this.activeNovelDisplay.querySelector('.novel-label').textContent = novel.title;
            this.queryInput.disabled = false;
            this.submitBtn.disabled = false;
            this.queryInput.placeholder = `Ask about "${novel.title}"...`;

            // Hide welcome, show ready state
            if (this.welcomeMessage) {
                this.welcomeMessage.remove();
            }
        } else {
            this.activeNovelDisplay.classList.remove('has-novel');
            this.activeNovelDisplay.querySelector('.novel-label').textContent = 'No book selected';
            this.queryInput.disabled = true;
            this.submitBtn.disabled = true;
            this.queryInput.placeholder = 'Select a book first...';
        }

        this.renderNovelsList();
    }

    async deleteNovel(novelId) {
        if (!confirm('Are you sure you want to delete this book?')) return;

        try {
            const response = await fetch(`/api/novels/${novelId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.novels = this.novels.filter(n => n.id !== novelId);
                if (this.activeNovel?.id === novelId) {
                    this.setActiveNovel(null);
                }
                this.renderNovelsList();
            } else {
                alert('Failed to delete novel');
            }
        } catch (error) {
            console.error('Failed to delete novel:', error);
            alert('Failed to delete novel');
        }
    }

    // === Upload ===

    handleFileSelect(e) {
        if (e.target.files.length) {
            this.handleFile(e.target.files[0]);
        }
    }

    handleFile(file) {
        const validTypes = ['.txt', '.pdf', '.epub'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();

        if (!validTypes.includes(ext)) {
            alert(`Invalid file type. Supported: ${validTypes.join(', ')}`);
            return;
        }

        this.pendingFile = file;
        this.bookTitle.value = file.name.replace(/\.[^/.]+$/, '');
        this.bookAuthor.value = 'Unknown';
        this.uploadModal.classList.add('open');
    }

    closeUploadModal() {
        this.uploadModal.classList.remove('open');
        this.pendingFile = null;
    }

    async handleUploadSubmit(e) {
        e.preventDefault();

        if (!this.pendingFile) return;

        const formData = new FormData();
        formData.append('file', this.pendingFile);
        formData.append('title', this.bookTitle.value || this.pendingFile.name);
        formData.append('author', this.bookAuthor.value || 'Unknown');

        this.closeUploadModal();
        this.showProcessing('Processing your book...', 'Uploading file...');

        try {
            const response = await fetch('/api/novels', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }

            const result = await response.json();

            // Simulate processing progress (actual progress would need SSE)
            for (let i = 0; i <= 100; i += 10) {
                this.updateProgress(i, `Indexing chapters...`);
                await this.delay(100);
            }

            this.hideProcessing();

            // Add to list and select
            this.novels.push(result.novel);
            this.renderNovelsList();
            await this.selectNovel(result.novel.id);

        } catch (error) {
            console.error('Upload failed:', error);
            this.hideProcessing();
            alert(`Upload failed: ${error.message}`);
        }

        this.pendingFile = null;
    }

    showProcessing(title, status) {
        this.processingTitle.textContent = title;
        this.processingStatus.textContent = status;
        this.progressFill.style.width = '0%';
        this.progressText.textContent = '0%';
        this.processingModal.classList.add('open');
    }

    updateProgress(percent, status) {
        this.progressFill.style.width = `${percent}%`;
        this.progressText.textContent = `${percent}%`;
        if (status) this.processingStatus.textContent = status;
    }

    hideProcessing() {
        this.processingModal.classList.remove('open');
    }

    // === Query ===

    async handleQuery(e) {
        e.preventDefault();

        const query = this.queryInput.value.trim();
        if (!query || !this.activeNovel || this.queryInFlight) return;

        // Add user message
        this.addMessage(query, 'user');
        this.queryInput.value = '';
        this.autoResizeTextarea();

        // Show loading
        this.setQueryPending(true);
        const loadingEl = this.addLoading();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, stream: false })
            });

            if (!response.ok) {
                throw new Error('Query failed');
            }

            const result = await response.json();

            // Remove loading
            loadingEl.remove();

            // Add response
            this.addMessage(result.answer, 'assistant', {
                refused: result.refused,
                refusalReason: result.refusal_reason,
                chapters: result.chapters_cited,
                sources: result.sources,
                timing: result.timing
            });

            // Update sources panel
            if (result.sources?.length) {
                this.showSources(result.sources);
            }

        } catch (error) {
            console.error('Query failed:', error);
            loadingEl.remove();
            this.addMessage('Sorry, something went wrong. Please try again.', 'assistant', { refused: true });
        } finally {
            this.setQueryPending(false);
        }
    }

    addMessage(content, type, options = {}) {
        const div = document.createElement('div');
        div.className = `message ${type}`;
        if (options.refused) div.classList.add('refused');

        if (type === 'assistant') {
            let html = `<div class="content">${this.escapeHtml(content)}</div>`;

            if (options.chapters?.length) {
                html += `
                    <div class="citations">
                        Sources: ${options.chapters.map(c => `Chapter ${c}`).join(', ')}
                        <button class="view-sources">View</button>
                    </div>
                `;
            }

            if (options.timing || options.refusalReason) {
                html += `<div class="message-meta">${this.formatMeta(options)}</div>`;
            }

            div.innerHTML = html;

            div.querySelector('.view-sources')?.addEventListener('click', () => {
                this.toggleSources(true);
            });
        } else {
            div.textContent = content;
        }

        this.messages.appendChild(div);
        this.messages.scrollTop = this.messages.scrollHeight;

        return div;
    }


    setQueryPending(pending) {
        this.queryInFlight = pending;
        this.queryInput.disabled = pending || !this.activeNovel;
        this.submitBtn.disabled = pending || !this.activeNovel;
    }

    formatMeta(options) {
        const parts = [];

        if (options.timing) {
            const totalMs = Object.values(options.timing).reduce((sum, sec) => sum + (sec * 1000), 0);
            if (Number.isFinite(totalMs) && totalMs > 0) {
                parts.push(`Latency: ${Math.round(totalMs)}ms`);
            }
        }

        if (options.refused && options.refusalReason) {
            parts.push(`Refusal: ${this.escapeHtml(options.refusalReason.replaceAll('_', ' '))}`);
        }

        return parts.join(' · ');
    }

    addLoading() {
        const div = document.createElement('div');
        div.className = 'message assistant loading';
        div.innerHTML = '<div class="loading"><span></span><span></span><span></span></div>';
        this.messages.appendChild(div);
        this.messages.scrollTop = this.messages.scrollHeight;
        return div;
    }

    // === Sources ===

    toggleSources(open = null) {
        if (open === null) {
            this.sourcesPanel.classList.toggle('open');
        } else if (open) {
            this.sourcesPanel.classList.add('open');
        } else {
            this.sourcesPanel.classList.remove('open');
        }
    }

    showSources(sources) {
        this.sourcesList.innerHTML = sources.map(source => `
            <div class="source-card">
                <div class="chapter">Chapter ${source.chapter_number}: ${this.escapeHtml(source.chapter_title || '')}</div>
                <div class="excerpt">${this.escapeHtml(source.content?.substring(0, 200) || '')}...</div>
            </div>
        `).join('');
    }

    // === Utilities ===

    autoResizeTextarea() {
        this.queryInput.style.height = 'auto';
        this.queryInput.style.height = Math.min(this.queryInput.scrollHeight, 200) + 'px';
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.novelRAG = new NovelRAG();
});
