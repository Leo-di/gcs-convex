"""Add presentation-friendly speed controls to a standalone MeshCat replay."""


_CONTROLS = """
<!-- meshcat-playback-speed-controls -->
<style>
  #replay-speed {
    position: fixed; bottom: 20px; left: 20px; z-index: 1000;
    display: flex; align-items: center; gap: 8px; padding: 10px 14px;
    border: 1px solid #cbd5e1; border-radius: 10px;
    background: rgba(255, 255, 255, .96); color: #172554;
    font: 15px system-ui, sans-serif;
    box-shadow: 0 2px 12px #0002;
  }
  #replay-speed button {
    min-width: 46px; padding: 7px 10px; border: 1px solid #94a3b8;
    border-radius: 6px; background: white; color: #172554;
    font: inherit; cursor: pointer;
  }
  #replay-speed button[aria-pressed="true"] {
    background: #1d4ed8; border-color: #1d4ed8; color: white;
  }
  #replay-speed button:disabled { opacity: .5; cursor: wait; }
  #replay-speed button:focus-visible { outline: 3px solid #60a5fa; }
</style>
<div id="replay-speed" role="group" aria-label="Playback speed">
  <span>Playback speed</span>
  <button type="button" data-rate="1" aria-pressed="true" disabled>1×</button>
  <button type="button" data-rate="2" aria-pressed="false" disabled>2×</button>
  <button type="button" data-rate="4" aria-pressed="false" disabled>4×</button>
</div>
<script>
(() => {
  const buttons = [...document.querySelectorAll('#replay-speed button')];
  const animator = viewer.animator;
  function sync() {
    const ready = animator.actions.length > 0;
    for (const button of buttons) {
      button.disabled = !ready;
      button.setAttribute('aria-pressed',
        String(Number(button.dataset.rate) === animator.mixer.timeScale));
    }
  }
  for (const button of buttons) {
    button.addEventListener('click', () => {
      animator.mixer.timeScale = Number(button.dataset.rate);
      sync();
    });
  }
  // Read the current mixer, including after MeshCat loads a new animation.
  // Pause, seek, and restart remain under MeshCat's native playback controls.
  setInterval(sync, 200);
  sync();
})();
</script>
"""


def add_playback_controls(page: str) -> str:
    """Keep the replay self-contained and preserve all recorded scene data."""
    if '<!-- meshcat-playback-speed-controls -->' in page:
        return page
    if '</body>' not in page:
        raise ValueError('Expected a complete MeshCat HTML page')
    return page.replace('</body>', _CONTROLS + '\n</body>', 1)
