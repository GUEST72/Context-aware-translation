import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Upload, 
  FileText, 
  Settings, 
  HelpCircle, 
  ChevronLeft, 
  ChevronRight, 
  Languages,
  History, 
  Star, 
  Edit3, 
  Archive,
  Search,
  BookOpen,
  Share2,
  Plus
} from 'lucide-react';
import { cn } from './lib/utils';

// --- Types ---

type ViewState = 'upload' | 'workspace';

interface TranslationHistoryItem {
  id: string;
  original: string;
  translated: string;
  timestamp: string;
  starred?: boolean;
}

// --- Components ---

const TopBar = ({ view, setView }: { view: ViewState; setView: (v: ViewState) => void }) => {
  return (
    <header className="fixed top-0 w-full z-50 bg-surface/70 backdrop-blur-xl border-b border-outline-variant/10 flex justify-between items-center px-8 h-16">
      <div className="flex items-center gap-8">
        <div 
          className="flex items-center gap-2 cursor-pointer" 
          onClick={() => setView('upload')}
        >
          <div className="w-8 h-8 primary-gradient rounded-lg flex items-center justify-center">
            <BookOpen className="text-on-primary w-5 h-5" />
          </div>
          <span className="text-xl font-bold tracking-tighter text-on-surface font-headline">LexisFlow</span>
        </div>
        
        <nav className="hidden md:flex gap-6 items-center">
          {[
            { id: 'documents', label: 'Documents', icon: FileText },
            { id: 'dictionary', label: 'Dictionary', icon: Search },
            { id: 'translate', label: 'Translate', icon: Languages },
          ].map((item) => (
            <button
              key={item.id}
              className={cn(
                "text-sm font-medium tracking-tight transition-all duration-300 flex items-center gap-2",
                view === 'workspace' && item.id === 'documents' 
                  ? "text-primary border-b-2 border-primary pb-1" 
                  : "text-on-surface-variant hover:text-primary"
              )}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="flex items-center gap-4">
        <button className="text-on-surface-variant hover:text-primary transition-colors p-2 rounded-full hover:bg-surface-container-highest">
          <Share2 className="w-5 h-5" />
        </button>
        <button className="text-on-surface-variant hover:text-primary transition-colors p-2 rounded-full hover:bg-surface-container-highest">
          <Settings className="w-5 h-5" />
        </button>
        <button className="text-on-surface-variant hover:text-primary transition-colors p-2 rounded-full hover:bg-surface-container-highest">
          <HelpCircle className="w-5 h-5" />
        </button>
        <div className="w-8 h-8 rounded-full border border-outline-variant/20 overflow-hidden ml-2">
          <img 
            src="https://picsum.photos/seed/user/100/100" 
            alt="User" 
            className="w-full h-full object-cover"
            referrerPolicy="no-referrer"
          />
        </div>
      </div>
    </header>
  );
};

const UploadScreen = ({ onUpload }: { onUpload: () => void }) => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="flex-1 flex flex-col items-center justify-center px-6 relative min-h-[calc(100vh-64px)]"
    >
      {/* Background Glows */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/5 blur-[120px] rounded-full -z-10" />
      <div className="absolute bottom-0 right-0 w-[400px] h-[400px] bg-secondary/5 blur-[100px] rounded-full -z-10" />

      <div className="w-full max-w-3xl flex flex-col items-center text-center">
        <div className="mb-12">
          <h1 className="text-5xl md:text-7xl font-headline font-extrabold tracking-tighter text-on-surface mb-6 leading-tight">
            Archive your <span className="text-primary italic">knowledge.</span>
          </h1>
          <p className="text-on-surface-variant max-w-lg mx-auto text-lg leading-relaxed">
            Transform your static PDFs into an interactive obsidian-grade reading experience with AI-powered annotations.
          </p>
        </div>

        <div className="w-full group cursor-pointer" onClick={onUpload}>
          <div className="relative p-1 rounded-2xl bg-gradient-to-b from-outline-variant/20 to-transparent transition-all duration-500 hover:from-primary/30">
            <div className="bg-surface-container-low border border-outline-variant/10 rounded-2xl p-12 md:p-24 flex flex-col items-center justify-center transition-all duration-500 group-hover:bg-surface-container-high/80 backdrop-blur-sm shadow-2xl">
              <div className="w-20 h-20 rounded-full bg-surface-container-highest flex items-center justify-center mb-8 shadow-inner ring-1 ring-outline-variant/20 group-hover:ring-primary/40 group-hover:bg-surface-bright transition-all">
                <Upload className="text-primary w-10 h-10 group-hover:scale-110 transition-transform duration-500" />
              </div>
              <h2 className="text-2xl font-headline font-bold text-on-surface mb-3">Upload a PDF</h2>
              <p className="text-on-surface-variant mb-10 text-sm tracking-wide uppercase">Drag and drop your file here or click below</p>
              <button className="primary-gradient text-on-primary font-headline font-bold px-10 py-4 rounded-xl flex items-center gap-3 transition-all duration-300 active:scale-95 shadow-[0_10px_20px_rgba(0,226,238,0.2)] hover:shadow-[0_15px_30px_rgba(0,226,238,0.3)]">
                <Plus className="w-5 h-5" />
                Select File
              </button>

              <div className="mt-12 flex items-center gap-6 opacity-40 group-hover:opacity-100 transition-opacity duration-700">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-widest font-medium">Encrypted</span>
                </div>
                <div className="w-1 h-1 bg-outline-variant rounded-full" />
                <div className="flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-widest font-medium">Fast Processing</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-16 w-full">
          {[
            { title: 'Precision Highlighting', desc: 'Extract core concepts with our surgical-grade annotation engine.', icon: Edit3, color: 'text-secondary' },
            { title: 'Tonal Theming', desc: 'Automated color-coding based on the semantic weight of your notes.', icon: Languages, color: 'text-primary' },
            { title: 'Fluid Export', icon: Archive, desc: 'Sync your highlights directly to Obsidian, Notion, or Markdown.', color: 'text-primary-dim' },
          ].map((feature, i) => (
            <div key={i} className="bg-surface-container-low p-6 rounded-xl border border-outline-variant/10 flex flex-col items-start text-left hover:bg-surface-container-high transition-colors group">
              <feature.icon className={cn("w-6 h-6 mb-4", feature.color)} />
              <h3 className="font-headline font-bold text-sm mb-2 text-on-surface">{feature.title}</h3>
              <p className="text-xs text-on-surface-variant leading-relaxed">{feature.desc}</p>
            </div>
          ))}
        </div>
      </div>
      
      <footer className="mt-20 py-12 flex flex-col items-center gap-6">
        <div className="flex gap-8 items-center text-xs text-on-surface-variant uppercase tracking-[0.2em] font-medium">
          <a href="#" className="hover:text-primary transition-colors">Documentation</a>
          <a href="#" className="hover:text-primary transition-colors">Privacy</a>
          <a href="#" className="hover:text-primary transition-colors">Changelog</a>
        </div>
        <p className="text-[10px] text-outline-variant tracking-widest font-medium">© 2026 LEXISFLOW SYSTEMS. ALL RIGHTS RESERVED.</p>
      </footer>
    </motion.div>
  );
};

const Workspace = () => {
  const history: TranslationHistoryItem[] = [
    { id: '1', original: 'The concept of the Shadow Architect...', translated: 'El concepto del Arquitecto de las Sombras se basa en el vacío...', timestamp: '2m ago', starred: true },
    { id: '2', original: 'Intentional asymmetry anchors the layout...', translated: 'La asimetría intencional ancla el diseño editorial...', timestamp: '14m ago' },
    { id: '3', original: 'Treats the screen as physical space...', translated: 'Trata la pantalla como un espacio físico definido por la luz...', timestamp: '1h ago' },
  ];

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex h-[calc(100vh-64px)] w-full overflow-hidden"
    >
      {/* Left Pane: PDF Viewer */}
      <section className="w-[70%] relative flex flex-col bg-surface overflow-hidden pdf-canvas border-r border-outline-variant/10">
        <div className="flex-1 overflow-auto p-12 flex justify-center items-start">
          <div className="w-[85%] max-w-4xl bg-white shadow-2xl p-20 min-h-[1200px] relative text-[#0e0e10]">
            <h1 className="text-3xl font-bold mb-8 font-serif">Architectural Theory in the Shadow Age</h1>
            <p className="text-lg leading-relaxed mb-6 font-serif">
              The discourse surrounding urban development has shifted towards a new paradigm. We call this <span className="bg-primary/20 px-1">The Shadow Architect</span> approach. This design system is built upon the concept of "The Shadow Architect." Unlike traditional dark modes that simply invert a white interface, this system treats the screen as a physical space defined by light and void.
            </p>
            <p className="text-lg leading-relaxed mb-6 font-serif">
              It moves away from the "boxy" nature of standard web apps toward a high-end editorial experience that feels curated, quiet, and powerful. The system achieves a premium feel through intentional asymmetry, where large typographic displays anchor the layout, and content "floats" within a deep, infinite canvas. We prioritize the "glow" over the line, using light as a functional tool rather than a decorative one.
            </p>

            {/* Floating Tooltip */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className="absolute top-[280px] left-[50%] -translate-x-1/2 bg-surface-container-highest text-primary px-4 py-2 rounded-xl shadow-2xl flex items-center gap-2 border border-primary/20 glass-hud"
            >
              <Languages className="w-4 h-4" />
              <span className="text-sm font-medium">Translate Selection</span>
            </motion.div>
          </div>
        </div>

        {/* Bottom Pagination HUD */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-8 z-50 glass-hud border border-outline-variant/20 rounded-full px-8 py-4 shadow-2xl">
          <button className="text-on-surface-variant hover:text-on-surface transition-all flex items-center gap-1 group">
            <ChevronLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
            <span className="text-xs font-medium">Previous</span>
          </button>
          <div className="text-primary font-bold text-xs flex items-center gap-2">
            <FileText className="w-4 h-4" />
            Page 12 of 45
          </div>
          <button className="text-on-surface-variant hover:text-on-surface transition-all flex items-center gap-1 group">
            <span className="text-xs font-medium">Next</span>
            <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </section>

      {/* Right Pane: Sidebar */}
      <aside className="w-[30%] bg-surface-container-low flex flex-col h-full border-l border-outline-variant/10">
        <div className="p-8 space-y-1">
          <div className="flex justify-between items-start mb-6">
            <div>
              <h2 className="text-lg font-black text-on-surface font-headline uppercase tracking-tight">Translation Engine</h2>
              <span className="font-headline uppercase tracking-widest text-[10px] text-primary">V2.4 Active</span>
            </div>
            <button className="primary-gradient text-on-primary px-4 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-widest active:scale-95 transition-transform">
              New Project
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-8 pb-8 space-y-10">
          {/* Latest Translation */}
          <section>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-headline uppercase tracking-widest text-[10px] text-on-surface-variant">Latest Translation</h3>
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse" />
            </div>
            <div className="bg-surface-container-highest/40 p-6 rounded-xl border border-outline-variant/10 space-y-4">
              <div className="h-3 w-3/4 bg-surface-container-high rounded-full skeleton-shimmer" />
              <div className="h-3 w-full bg-surface-container-high rounded-full skeleton-shimmer" />
              <div className="h-3 w-5/6 bg-surface-container-high rounded-full skeleton-shimmer" />
              <div className="pt-4 flex justify-between items-center">
                <div className="flex gap-2">
                  <div className="w-8 h-2 bg-primary/20 rounded-full" />
                  <div className="w-12 h-2 bg-primary/20 rounded-full" />
                </div>
                <span className="text-[10px] text-primary font-bold">TRANSLATING...</span>
              </div>
            </div>
          </section>

          {/* History */}
          <section className="flex-1">
            <h3 className="font-headline uppercase tracking-widest text-[10px] text-on-surface-variant mb-6">Translation History</h3>
            <div className="space-y-6">
              {history.map((item) => (
                <div key={item.id} className="group cursor-pointer">
                  <div className="p-4 rounded-xl transition-all duration-300 hover:bg-surface-container-highest border border-transparent hover:border-outline-variant/20">
                    <p className="text-xs text-on-surface-variant italic mb-2 line-clamp-1">"{item.original}"</p>
                    <p className="text-sm text-on-surface font-medium leading-relaxed">{item.translated}</p>
                    <div className="mt-3 flex items-center gap-3 text-[10px] text-on-surface-variant">
                      <span className="flex items-center gap-1"><History className="w-3 h-3" /> {item.timestamp}</span>
                      {item.starred && (
                        <span className="flex items-center gap-1 text-primary"><Star className="w-3 h-3 fill-primary" /> Starred</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Sidebar Footer */}
        <div className="p-6 border-t border-outline-variant/10 grid grid-cols-4 gap-2">
          {[
            { label: 'History', icon: History, active: false },
            { label: 'Starred', icon: Star, active: true },
            { label: 'Drafts', icon: Edit3, active: false },
            { label: 'Archive', icon: Archive, active: false },
          ].map((nav) => (
            <button 
              key={nav.label}
              className={cn(
                "flex flex-col items-center gap-1 p-2 rounded-md transition-all duration-300",
                nav.active ? "primary-gradient text-on-primary" : "text-on-surface-variant hover:bg-surface-container-highest"
              )}
            >
              <nav.icon className={cn("w-5 h-5", nav.active ? "fill-on-primary" : "")} />
              <span className="font-headline uppercase tracking-widest text-[8px] font-bold">{nav.label}</span>
            </button>
          ))}
        </div>
      </aside>
    </motion.div>
  );
};

export default function App() {
  const [view, setView] = React.useState<ViewState>('upload');

  return (
    <div className="min-h-screen bg-surface selection:bg-primary/30 selection:text-primary">
      <TopBar view={view} setView={setView} />
      <main className="pt-16">
        <AnimatePresence mode="wait">
          {view === 'upload' ? (
            <div key="upload">
              <UploadScreen onUpload={() => setView('workspace')} />
            </div>
          ) : (
            <div key="workspace">
              <Workspace />
            </div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
