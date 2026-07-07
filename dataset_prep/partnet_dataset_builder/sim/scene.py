import os
try:
    import sapien.core as sapien
except ImportError:
    import sapien

def create_scene():
    # Force offscreen to prevent windowing errors
    os.environ["SAP_USE_EGL"] = "1"
    
    if hasattr(sapien, "Engine"):
        # SAPIEN 2.x
        engine = sapien.Engine()
        renderer = sapien.SapienRenderer(offscreen_only=True)
        engine.set_renderer(renderer)
        scene = engine.create_scene()
    else:
        # SAPIEN 3.x
        scene = sapien.Scene()
        try:
            from sapien.physx import PhysxSystem
            from sapien.render import RenderSystem
            scene.add_system(PhysxSystem())
            scene.add_system(RenderSystem())
        except ImportError: pass

    scene.set_timestep(1 / 240.0)

    # --- BALANCED LIGHTING SETUP ---
    # Moderate ambient light (not too bright, not too dark)
    scene.set_ambient_light([0.4, 0.4, 0.4]) 
    
    # Main key light from above-front
    scene.add_directional_light([0, 0.5, -1], [0.8, 0.8, 0.8], shadow=True)
    # Fill light from opposite side
    scene.add_directional_light([0, -0.5, -0.5], [0.4, 0.4, 0.4], shadow=False)
    # Rim light from behind
    scene.add_directional_light([-1, 0, 0], [0.3, 0.3, 0.3], shadow=False)

    return scene