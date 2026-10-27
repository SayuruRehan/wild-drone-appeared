"""Exact simulator rendering; fire and inspection freshness are separate views."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_frame(env):
    width, height, panel = 980, 640, 430
    image = Image.new('RGB', (width, height), '#0e1728')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 15)
        title_font = ImageFont.truetype('DejaVuSans.ttf', 25)
    except OSError:
        font = title_font = ImageFont.load_default()
    draw.text((30, 18), 'WILD DRONE APPEARED', fill='#e6edf6', font=title_font)
    wind_labels = ['calm', 'north', 'south', 'west', 'east']
    draw.text((30, 57), f'Fully observable fleet MDP  |  step {env.elapsed}/'
              f'{env.config.horizon}  |  wind: {wind_labels[env.wind]}',
              fill='#9eabc2', font=font)
    draw.text((30, 86), 'FIRE, FUEL & PRIORITY ZONES', fill='#e6edf6', font=font)
    draw.text((520, 86), 'INSPECTION FRESHNESS', fill='#e6edf6', font=font)
    fuel = env.fuel/env.config.max_fuel
    fire = np.zeros((*env.fuel.shape, 3), dtype=np.uint8)
    fire[:, :, 0] = (25+25*fuel).astype(np.uint8)
    fire[:, :, 1] = (38+100*fuel).astype(np.uint8)
    fire[:, :, 2] = (48+25*fuel).astype(np.uint8)
    fire[env.fuel == 0] = [37, 42, 52]
    fire[env.priority > 1] = np.clip(fire[env.priority > 1].astype(int)+[35, 5, 30], 0, 255)
    fire[env.burning] = [255, 97, 45]
    freshness = 1-np.minimum(env.ages/env.config.inspection_deadline, 1)
    fresh = np.stack([45+20*freshness, 50+170*freshness,
                      80+110*freshness], axis=-1).astype(np.uint8)
    colours = ['#4cd9fa', '#ffd45a', '#da90ff']
    for left, array in [(30, fire), (520, fresh)]:
        img = Image.fromarray(array).resize((panel, panel), Image.Resampling.NEAREST)
        image.paste(img, (left, 113))
        draw.rectangle((left, 113, left+panel, 113+panel), outline='#596781')
        unit = panel/env.config.grid_size
        base_y, base_x = env.base
        x, y = left+(base_x+0.5)*unit, 113+(base_y+0.5)*unit
        draw.rectangle((x-6, y-6, x+6, y+6), fill='#ffffff')
        draw.text((x+8, y-10), 'BASE', fill='#ffffff', font=font)
        for i, (row, col) in enumerate(env.positions):
            x, y = left+(col+0.5)*unit, 113+(row+0.5)*unit
            # Slight label offsets keep parked drone identities legible.
            colour = '#64748b' if env.lost[i] else colours[i]
            draw.ellipse((x-5, y-5, x+5, y+5), fill=colour, outline='#0e1728')
            draw.text((x+7, y-10+14*i), str(i+1), fill=colour, font=font)
    draw.text((30, 556), 'Orange: fire   Purple tint: priority   White square: base',
              fill='#9eabc2', font=font)
    draw.text((520, 556), 'Bright: fresh inspection   Dark: stale / overdue',
              fill='#9eabc2', font=font)
    for i, colour in enumerate(colours):
        x = 30+315*i
        status = 'LOST' if env.lost[i] else ('CHARGING' if env.charger_owner == i else 'ACTIVE')
        draw.text((x, 588), f'Drone {i+1}: {env.batteries[i]:.0f}/'
                  f'{env.config.battery_capacity:.0f}  {status}', fill=colour, font=font)
        draw.rectangle((x, 614, x+260, 622), fill='#27334b')
        draw.rectangle((x, 614, x+260*env.batteries[i]/env.config.battery_capacity, 622),
                       fill=colour)
    return np.array(image)
