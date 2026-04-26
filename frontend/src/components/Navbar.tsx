/**
 * Navbar — Sticky top navigation with section links.
 */

import type { ActiveSection } from '../types';

const SECTIONS: { id: ActiveSection; label: string }[] = [
  { id: 'playground', label: 'Playground' },
  { id: 'pipeline', label: 'Architecture' },
  { id: 'dashboard', label: 'Dashboard' },
];

interface Props {
  activeSection: ActiveSection;
  onSectionChange: (s: ActiveSection) => void;
}

export function Navbar({ activeSection, onSectionChange }: Props) {
  const scrollTo = (id: ActiveSection) => {
    onSectionChange(id);
  };

  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <div className="navbar__brand">
        <div className="navbar__icon" aria-hidden="true">⚡</div>
        <span className="navbar__title">GNN RecSys</span>
      </div>

      <div className="navbar__links">
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            className={`navbar__link ${activeSection === s.id ? 'navbar__link--active' : ''}`}
            onClick={() => scrollTo(s.id)}
            aria-current={activeSection === s.id ? 'page' : undefined}
          >
            {s.label}
          </button>
        ))}
      </div>
    </nav>
  );
}
