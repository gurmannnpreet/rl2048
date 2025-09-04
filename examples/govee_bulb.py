# Example integration: control a Govee bulb.
# Requires `govee_api_laggat`. Set GOVEE_API_KEY in your environment.

import os
import asyncio
from govee_api_laggat import Govee  # type: ignore

API_KEY = os.getenv("GOVEE_API_KEY")


def _is_bulb(d):
    get = (lambda k: (d.get(k) if isinstance(d, dict) else getattr(d, k, None)) or "")
    model = str(get("model")).lower()
    name = str(get("device_name")).lower()
    t = str(get("deviceType")).lower()
    return ("bulb" in model) or ("bulb" in name) or ("light" in t)


async def main():
    if not API_KEY:
        raise SystemExit("Set GOVEE_API_KEY env var first")

    govee = await Govee.create(api_key=API_KEY)
    devices = await govee.get_devices()
    bulbs = [d for d in devices if _is_bulb(d)]
    if not bulbs:
        print("No Govee bulbs found.")
        await govee.close()
        return

    bulb = bulbs[0]
    print("Found bulb:", (getattr(bulb, "device_name", None) or getattr(bulb, "name", None) or "unknown"))

    try:
        state = await govee.get_state(bulb)
        print("Current state:", state)
    except Exception as e:
        print("Could not get state (continuing):", e)

    await govee.turn_on(bulb)
    await govee.set_color(bulb, (0, 0, 255))  # RGB
    await govee.set_brightness(bulb, 50)
    await govee.turn_off(bulb)

    await govee.close()


if __name__ == "__main__":
    asyncio.run(main())

