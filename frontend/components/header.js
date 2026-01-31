// Common Header Component for NeuroQuant
function createHeader(activePage) {
    return `
    <header class="border-b border-neutral-900 bg-black/50 backdrop-blur-xl sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-8">
                <h1 class="text-xl font-semibold">NeuroQuant</h1>
                <nav class="flex items-center space-x-6 text-sm">
                    <a href="/" class="${activePage === 'home' ? 'text-white font-medium' : 'text-neutral-400 hover:text-white transition'}">Home</a>
                    <a href="/dashboard" class="${activePage === 'dashboard' ? 'text-white font-medium' : 'text-neutral-400 hover:text-white transition'}">Dashboard</a>
                    <a href="/strategies" class="${activePage === 'strategies' ? 'text-white font-medium' : 'text-neutral-400 hover:text-white transition'}">Strategies</a>
                    <a href="/backtest" class="${activePage === 'backtest' ? 'text-white font-medium' : 'text-neutral-400 hover:text-white transition'}">Backtest</a>
                </nav>
            </div>
            <div class="flex items-center space-x-2 px-3 py-1.5 bg-neutral-900 rounded-lg">
                <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span class="text-xs font-mono text-neutral-400">LIVE</span>
            </div>
        </div>
    </header>
    `;
}

function createFooter() {
    return `
    <footer class="border-t border-neutral-900 mt-16 py-8">
        <div class="max-w-7xl mx-auto px-6 text-center text-sm text-neutral-500">
            <p>NeuroQuant © ${new Date().getFullYear()} • Algorithmic Trading Platform</p>
        </div>
    </footer>
    `;
}

// Export for use in pages
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createHeader, createFooter };
}
