export function ChatPanel() {
  return (
    <section className="workspace-panel">
      <h2 className="workspace-panel__title">Chat</h2>
      <div className="workspace-panel__body workspace-panel__body--centered">
        <div className="chat-skeleton">
          <div className="chat-skeleton__line chat-skeleton__line--long" />
          <div className="chat-skeleton__line" />
          <div className="chat-skeleton__line chat-skeleton__line--short" />
        </div>
        <p className="page-lede">List-scoped chat arrives in v1. This panel stays intentionally quiet until then.</p>
      </div>
    </section>
  );
}
