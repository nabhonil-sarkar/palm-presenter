"""Run Palm Presenter on this computer and open its browser interface."""

import argparse
import json
import time
import webbrowser
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local port (default: choose an available port)",
    )
    parser.add_argument(
        "--model-dir", type=Path, help="Use a different directory for model downloads"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the session URL without opening a tab",
    )
    options = parser.parse_args()
    if not 0 <= options.port <= 65535:
        parser.error("--port must be between 0 and 65535")

    from application import PresenterApplication, launch_interface
    from vision import VisionModels, model_assets

    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
    print(
        "Preparing recognition models. The first run downloads about 100 MB.",
        flush=True,
    )
    models = VisionModels(model_assets(manifest, options.model_dir), tracking=True)
    server = None
    thread = None
    try:
        server, thread, url = launch_interface(
            PresenterApplication(models),
            (root / "interface.html").read_text(encoding="utf-8"),
            port=options.port,
        )
        print(
            f"Open this local session in your browser:\n{url}\nPress Ctrl+C here to exit.",
            flush=True,
        )
        if not options.no_browser:
            webbrowser.open_new_tab(url)
        while thread.is_alive() and not server.should_exit:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.should_exit = True
            thread.join()
        models.close()


if __name__ == "__main__":
    main()
