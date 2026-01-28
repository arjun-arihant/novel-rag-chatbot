// Novel RAG Chatbot - Frontend JavaScript

class NovelRAGApp {
    constructor() {
        this.elements = {
            queryForm: document.getElementById('queryForm'),
            queryInput: document.getElementById('queryInput'),
            submitBtn: document.getElementById('submitBtn'),
            messages: document.getElementById('messages'),
            status: document.getElementById('status'),
            sidebar: document.getElementById('sidebar'),
            sourcesList: document.getElementById('sourcesList'),
            closeSidebar: document.getElementById('closeSidebar'),
            ingestModal: document.getElementById('ingestModal'),
            novelPath: document.getElementById('novelPath'),
            ingestBtn: document.getElementById('ingestBtn'),
            cancelIngest: document.getElementById('cancelIngest'),
            ingestStatus: document.getElementById('ingestStatus'),
        };

        this.isLoading = false;
        this.currentSources = [];
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkHealth();
        this.autoResizeTextarea();
    }

    bindEvents() {
        // Query form
        this.elements.queryForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitQuery();
        });

        // Auto-resize textarea
        this.elements.queryInput.addEventListener('input', () => {
            this.autoResizeTextarea();
        });

        // Enter to submit (Shift+Enter for newline)
        this.elements.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.submitQuery();
            }
        });

        // Sidebar
        this.elements.closeSidebar.addEventListener('click', () => {
            this.elements.sidebar.classList.remove('open');
        });

        // Ingest modal
        this.elements.ingestBtn.addEventListener('click', () => this.ingestNovel());
        this.elements.cancelIngest.addEventListener('click', () => {
            this.elements.ingestModal.classList.remove('open');
        });
    }

    autoResizeTextarea() {
        const textarea = this.elements.queryInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (data.pipeline_ready) {
                this.setStatus('ready', 'Ready');
            } else {
                this.setStatus('warning', 'No novel loaded');
                this.elements.ingestModal.classList.add('open');
            }
        } catch (error) {
            this.setStatus('error', 'Connection failed');
        }
    }

    setStatus(state, text) {
        this.elements.status.className = 'status ' + state;
        this.elements.status.querySelector('.status-text').textContent = text;
    }

    async ingestNovel() {
        const path = this.elements.novelPath.value.trim();
        if (!path) return;

        this.elements.ingestBtn.disabled = true;
        this.elements.ingestStatus.textContent = 'Loading novel...';
        this.elements.ingestStatus.className = 'ingest-status';

        try {
            const response = await fetch('/api/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ novel_path: path, force_reindex: false })
            });

            const data = await response.json();

            if (response.ok) {
                this.elements.ingestStatus.textContent = 
                    `Loaded ${data.chapters || 0} chapters, ${data.chunks || 0} chunks`;
                this.elements.ingestStatus.className = 'ingest-status success';
                this.setStatus('ready', 'Ready');
                
                setTimeout(() => {
                    this.elements.ingestModal.classList.remove('open');
                }, 1500);
            } else {
                throw new Error(data.detail || 'Ingestion failed');
            }
        } catch (error) {
            this.elements.ingestStatus.textContent = error.message;
            this.elements.ingestStatus.className = 'ingest-status error';
        } finally {
            this.elements.ingestBtn.disabled = false;
        }
    }

    async submitQuery() {
        const query = this.elements.queryInput.value.trim();
        if (!query || this.isLoading) return;

        this.isLoading = true;
        this.elements.submitBtn.disabled = true;
        this.elements.queryInput.value = '';
        this.autoResizeTextarea();

        // Clear welcome message
        const welcome = this.elements.messages.querySelector('.welcome-message');
        if (welcome) welcome.remove();

        // Add user message
        this.addMessage(query, 'user');

        // Add loading indicator
        const loadingEl = this.addLoading();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, stream: false })
            });

            const data = await response.json();

            if (response.ok) {
                loadingEl.remove();
                this.addAssistantMessage(data);
                this.currentSources = data.sources || [];
                this.updateSources();
            } else {
                throw new Error(data.detail || 'Query failed');
            }
        } catch (error) {
            loadingEl.remove();
            this.addMessage(`Error: ${error.message}`, 'assistant', true);
        } finally {
            this.isLoading = false;
            this.elements.submitBtn.disabled = false;
            this.elements.queryInput.focus();
        }
    }

    addMessage(content, type, isError = false) {
        const div = document.createElement('div');
        div.className = `message ${type}` + (isError ? ' refused' : '');
        div.innerHTML = `<div class="content">${this.escapeHtml(content)}</div>`;
        this.elements.messages.appendChild(div);
        this.scrollToBottom();
        return div;
    }

    addAssistantMessage(data) {
        const div = document.createElement('div');
        div.className = 'message assistant' + (data.refused ? ' refused' : '');
        
        let html = `<div class="content">${this.formatAnswer(data.answer)}</div>`;
        
        if (data.chapters_cited && data.chapters_cited.length > 0) {
            html += `
                <div class="citations">
                    Based on: ${data.chapters_cited.map(c => `Chapter ${c}`).join(', ')}
                    <button onclick="app.showSources()">View sources</button>
                </div>
            `;
        }
        
        div.innerHTML = html;
        this.elements.messages.appendChild(div);
        this.scrollToBottom();
    }

    addLoading() {
        const div = document.createElement('div');
        div.className = 'message assistant loading';
        div.innerHTML = '<span></span><span></span><span></span>';
        this.elements.messages.appendChild(div);
        this.scrollToBottom();
        return div;
    }

    showSources() {
        this.elements.sidebar.classList.add('open');
    }

    updateSources() {
        if (!this.currentSources.length) {
            this.elements.sourcesList.innerHTML = '<p class="no-sources">No sources available</p>';
            return;
        }

        this.elements.sourcesList.innerHTML = this.currentSources.map(source => `
            <div class="source-card">
                <div class="chapter">Chapter ${source.chapter_number}: ${this.escapeHtml(source.chapter_title)}</div>
                <div class="excerpt">${this.escapeHtml(source.content)}</div>
            </div>
        `).join('');
    }

    scrollToBottom() {
        this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatAnswer(text) {
        // Convert [Chapter X] citations to styled spans
        return this.escapeHtml(text).replace(
            /\[Chapter\s+(\d+)\]/gi,
            '<span class="citation">[Chapter $1]</span>'
        );
    }
}

// Initialize app
const app = new NovelRAGApp();
