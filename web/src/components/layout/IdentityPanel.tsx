import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { QrCode, LogOut } from "lucide-react";

import { useLanguage } from "@/app/LanguageContext";

type IdentityPanelProps = {
  bilibiliCookie: string;
  bilibiliUsername: string;
  bilibiliAvatarUrl: string;
  onClose: () => void;
  onLogin: () => void;
  onLogout: () => void;
};

const PLATFORMS = [{ key: "bilibili", label: "Bilibili", avatarFallback: "B" }];

export default function IdentityPanel({
  bilibiliCookie,
  bilibiliUsername,
  bilibiliAvatarUrl,
  onClose,
  onLogin,
  onLogout,
}: IdentityPanelProps) {
  const { t } = useLanguage();
  const isSignedIn = bilibiliCookie.length > 0;
  const [avatarError, setAvatarError] = useState(false);

  useEffect(() => {
    setAvatarError(false);
  }, [bilibiliAvatarUrl]);

  return createPortal(
    <>
      <button
        type="button"
        className="fixed inset-0 z-float border-0 bg-transparent"
        aria-label={t("common.close")}
        onClick={onClose}
      />
      <div
        className="fixed right-4 top-14 z-float min-w-56 rounded-xl border border-border bg-white/96 p-3 shadow-lg backdrop-blur-md md:right-6 xl:right-12"
        role="menu"
        aria-label={t("navbar.identity")}
      >
        <div className="flex flex-col gap-1">
          {PLATFORMS.map((platform) => {
            const hasAvatar = isSignedIn && bilibiliAvatarUrl && !avatarError;
            return (
              <div
                key={platform.key}
                data-testid="bilibili-row"
                className="flex items-center gap-3 rounded-lg px-2 py-2"
              >
                <div className="relative flex shrink-0">
                  {hasAvatar ? (
                    <img
                      src={`/api/proxy/cover?url=${encodeURIComponent(bilibiliAvatarUrl)}`}
                      alt={bilibiliUsername || "Bilibili avatar"}
                      className="size-9 rounded-full object-cover"
                      onError={() => setAvatarError(true)}
                    />
                  ) : (
                    <span
                      className="inline-flex h-9 w-9 items-center justify-center rounded-full bg-pink/10 font-bold text-blue"
                      aria-hidden="true"
                    >
                      {platform.avatarFallback}
                    </span>
                  )}
                  <span
                    className={`absolute -bottom-0.5 -right-0.5 size-3 rounded-full border-2 border-white ${
                      isSignedIn ? "bg-green-400" : "bg-gray-300"
                    }`}
                    aria-label={isSignedIn ? t("navbar.signedIn") : t("navbar.signedOut")}
                  />
                </div>

                <div className="flex min-w-0 flex-1 flex-col">
                  <span className="text-sm font-semibold text-ink">{platform.label}</span>
                  {isSignedIn && bilibiliUsername ? (
                    <span className="truncate text-xs text-muted" title={bilibiliUsername}>
                      {bilibiliUsername}
                    </span>
                  ) : (
                    <span className="text-xs text-muted">
                      {isSignedIn ? t("navbar.signedIn") : t("navbar.signedOut")}
                    </span>
                  )}
                </div>

                <button
                  type="button"
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-muted transition hover:bg-sky/10 hover:text-ink"
                  aria-label={isSignedIn ? t("auth.bilibili.signOut") : t("navbar.signIn")}
                  onClick={() => (isSignedIn ? onLogout() : onLogin())}
                >
                  {isSignedIn ? (
                    <LogOut className="size-4" />
                  ) : (
                    <QrCode className="size-4" />
                  )}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </>,
    document.body,
  );
}
