import type { BibilabList } from "@/lib/types";
import { useLanguage } from "@/app/LanguageContext";
import { Button, Modal } from "@/components/ui";

interface DeleteListModalProps {
  list: BibilabList | null;
  open: boolean;
  onClose: () => void;
  onConfirm: (list: BibilabList) => Promise<void>;
}

export function DeleteListModal({ list, open, onClose, onConfirm }: DeleteListModalProps) {
  const { t } = useLanguage();

  return (
    <Modal
      footer={
        <>
          <Button onClick={onClose} size="sm" variant="ghost">
            {t("common.cancel")}
          </Button>
          <Button
            onClick={() => {
              if (list) {
                void onConfirm(list).then(() => onClose());
              }
            }}
            size="sm"
            variant="danger"
          >
            {t("common.delete")}
          </Button>
        </>
      }
      onClose={onClose}
      open={open}
      size="lg"
      title={t("home.deleteList")}
    >
      <div className="rounded-2xl border border-error/40 bg-error/10 p-4 text-sm text-error">
        <p className="m-0 text-base font-semibold tracking-tight">{t("home.cannotUndo")}</p>
        <p className="mt-1.5 mb-0 leading-6">
          {list
            ? t("home.deleteConfirm", { name: list.name, count: list.source_count })
            : ""}
        </p>
      </div>
    </Modal>
  );
}
