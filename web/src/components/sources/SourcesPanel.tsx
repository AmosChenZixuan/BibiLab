import type { NoteContent, Source } from "../../lib/types";
import { workspacePanelClass, workspacePanelTitleClass } from "../../lib/ui";

import { SourceDetail } from "./SourceDetail";
import { SourceList } from "./SourceList";

type Props = {
  activeTab: "note" | "transcript";
  detailSource: Source | null;
  ingestBusy: boolean;
  ingestError: string | null;
  ingestStatus: string | null;
  note: NoteContent | null;
  onBack: () => void;
  onDelete: (source: Source) => Promise<void>;
  onIngest: (url: string, rerun: boolean) => Promise<void>;
  onOpen: (source: Source) => Promise<void>;
  onSelectTab: (tab: "note" | "transcript") => Promise<void>;
  sources: Source[];
  transcript: string | null;
  transcriptError: string | null;
  transcriptLoading: boolean;
};

export function SourcesPanel(props: Props) {
  return (
    <section className={workspacePanelClass}>
      <h2 className={workspacePanelTitleClass}>Sources</h2>
      {props.detailSource ? (
        <SourceDetail
          activeTab={props.activeTab}
          note={props.note}
          onBack={props.onBack}
          onSelectTab={props.onSelectTab}
          source={props.detailSource}
          transcript={props.transcript}
          transcriptError={props.transcriptError}
          transcriptLoading={props.transcriptLoading}
        />
      ) : (
        <SourceList
          busy={props.ingestBusy}
          error={props.ingestError}
          ingestStatus={props.ingestStatus}
          onDelete={props.onDelete}
          onIngest={props.onIngest}
          onOpen={(source) => void props.onOpen(source)}
          sources={props.sources}
        />
      )}
    </section>
  );
}
