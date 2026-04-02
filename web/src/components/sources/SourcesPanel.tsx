import type { NoteContent, Source } from "../../lib/types";
import type { JobActivityItem } from "../jobs/JobActivityProvider";
import { Panel, PanelTitle } from "../../components/ui";

import { SourceDetail } from "./SourceDetail";
import { SourceList } from "./SourceList";

type Props = {
  activeTab: "note" | "transcript";
  detailSource: Source | null;
  ingestBusy: boolean;
  ingestError: string | null;
  ingestJobs: JobActivityItem[];
  ingestStatus: string | null;
  onDismissIngestJob: (jobId: string) => void;
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
    <Panel variant="workspace">
      <PanelTitle>Sources</PanelTitle>
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
          ingestJobs={props.ingestJobs}
          ingestStatus={props.ingestStatus}
          onDismissIngestJob={props.onDismissIngestJob}
          onDelete={props.onDelete}
          onIngest={props.onIngest}
          onOpen={(source) => void props.onOpen(source)}
          sources={props.sources}
        />
      )}
    </Panel>
  );
}
