import { useEffect, useRef, useState } from "react";

import { QRCodeSVG } from "qrcode.react";
import { useLanguage } from "@/app/LanguageContext";
import { Button, Modal } from "@/components/ui";
import { api, type BilibiliQrStatus } from "@/lib/api";

type BilibiliQrModalProps = {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
};

const POLL_INTERVAL_MS = 2000;

export function BilibiliQrModal({ open, onClose, onSuccess }: BilibiliQrModalProps) {
  const { t } = useLanguage();
  const [qr, setQr] = useState<{ url: string; key: string } | null>(null);
  const [status, setStatus] = useState<BilibiliQrStatus>("waiting");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const cancelled = useRef(false);

  useEffect(() => {
    if (!open) return;

    cancelled.current = false;
    setStatus("waiting");
    setError(null);
    setQr(null);
    setLoading(true);

    api.auth
      .generateBilibiliQr()
      .then((data) => {
        if (cancelled.current || !data) return;
        setQr(data);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled.current) return;
        setError(err instanceof Error ? err.message : "Failed to generate QR code");
        setLoading(false);
      });

    return () => {
      cancelled.current = true;
      if (pollTimer.current) {
        clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [open]);

  useEffect(() => {
    if (!open || !qr) return;

    pollTimer.current = setInterval(() => {
      if (cancelled.current) return;
      api.auth
        .pollBilibiliQr(qr.key)
        .then((data) => {
          if (cancelled.current || !data) return;
          setStatus(data.status);
          if (data.status === "success") {
            if (pollTimer.current) clearInterval(pollTimer.current);
            onSuccess();
          }
        })
        .catch(() => {
          // Silently ignore polling errors; will retry
        });
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollTimer.current) {
        clearInterval(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [open, qr, onSuccess]);

  function handleClose() {
    cancelled.current = true;
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
    onClose();
  }

  async function handleSignOut() {
    try {
      await api.auth.deleteBilibiliAuth();
      onSuccess();
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to sign out");
    }
  }

  const statusMessage = (() => {
    switch (status) {
      case "waiting":
        return t("auth.bilibili.waiting");
      case "scanned":
        return t("auth.bilibili.scanned");
      case "expired":
        return t("auth.bilibili.expired");
      case "success":
        return t("auth.bilibili.success");
    }
  })();

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title={t("auth.bilibili.title")}
      size="md"
      footer={
        <div className="flex w-full justify-between">
          <Button variant="ghost" size="sm" onClick={handleSignOut}>
            {t("auth.bilibili.signOut")}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleClose}>
            {t("common.cancel")}
          </Button>
        </div>
      }
    >
      <div className="flex flex-col items-center gap-4">
        {loading && (
          <div className="flex h-48 w-48 items-center justify-center">
            <span className="text-muted">{t("common.creating")}</span>
          </div>
        )}
        {error && (
          <div className="flex h-48 w-48 flex-col items-center justify-center gap-2 text-center">
            <span className="text-sm text-red">{error}</span>
          </div>
        )}
        {!loading && !error && qr && (
          <>
            <div className="relative">
              <QRCodeSVG value={qr.url} size={192} level="M" />
              {status === "expired" && (
                <div className="absolute inset-0 flex items-center justify-center bg-white/80">
                  <span className="text-sm font-semibold text-red">{t("auth.bilibili.expiredTip")}</span>
                </div>
              )}
            </div>
            <p
              className={`text-center text-sm ${
                status === "expired" ? "text-red" : status === "success" ? "text-green" : "text-muted"
              }`}
            >
              {statusMessage}
            </p>
          </>
        )}
      </div>
    </Modal>
  );
}
