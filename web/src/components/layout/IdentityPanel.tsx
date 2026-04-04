import { createPortal } from "react-dom";

type IdentityPanelProps = {
  onClose: () => void;
};

const PLATFORMS = [{ key: "bilibili", label: "Bilibili", icon: "B" }];

export default function IdentityPanel({ onClose }: IdentityPanelProps) {
  return createPortal(
    <>
      <button
        type="button"
        className="fixed inset-0 z-float border-0 bg-transparent"
        aria-label="Close identity panel"
        onClick={onClose}
      />
      <div
        className="fixed right-4 top-14 z-float min-w-44 rounded-xl border border-border bg-white/96 p-4 shadow-lg backdrop-blur-md md:right-6 xl:right-12"
        role="menu"
        aria-label="Identity"
      >
        <div className="flex flex-wrap gap-3">
          {PLATFORMS.map((platform) => (
            <div key={platform.key} className="flex w-18 flex-col items-center gap-1">
              <span
                className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-pink-50 font-bold text-blue"
                aria-hidden="true"
              >
                {platform.icon}
              </span>
              <span className="text-xs font-semibold text-ink">{platform.label}</span>
              <span className="text-center text-xs text-muted">Not signed in</span>
            </div>
          ))}
        </div>
      </div>
    </>,
    document.body,
  );
}
