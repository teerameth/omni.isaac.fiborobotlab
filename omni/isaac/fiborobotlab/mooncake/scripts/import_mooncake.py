# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import torch
from torch import roll
import omni
import omni.kit.commands
import omni.usd
import omni.client
import asyncio
import math
import weakref
import omni.ui as ui
from omni.kit.menu.utils import add_menu_items, remove_menu_items, MenuItemDescription
from omni.isaac.isaac_sensor import _isaac_sensor
from omni.isaac.core.prims import RigidPrimView

from .common import set_drive_parameters
from pxr import UsdLux, Sdf, Gf, UsdPhysics, Usd, UsdGeom

from omni.isaac.ui.ui_utils import setup_ui_headers, get_style, btn_builder
from omni.isaac.core.utils.prims import get_prim_at_path

EXTENSION_NAME = "Import Mooncake"
def create_parent_xforms(asset_usd_path, source_prim_path, save_as_path=None):
    """ Adds a new UsdGeom.Xform prim for each Mesh/Geometry prim under source_prim_path.
        Moves material assignment to new parent prim if any exists on the Mesh/Geometry prim.

        Args:
            asset_usd_path (str): USD file path for asset
            source_prim_path (str): USD path of root prim
            save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
    """
    omni.usd.get_context().open_stage(asset_usd_path)
    stage = omni.usd.get_context().get_stage()

    prims = [stage.GetPrimAtPath(source_prim_path)]
    edits = Sdf.BatchNamespaceEdit()
    while len(prims) > 0:
        prim = prims.pop(0)
        print(prim)
        if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
            new_xform = UsdGeom.Xform.Define(stage, str(prim.GetPath()) + "_xform")
            print(prim, new_xform)
            edits.Add(Sdf.NamespaceEdit.Reparent(prim.GetPath(), new_xform.GetPath(), 0))
            continue

        children_prims = prim.GetChildren()
        prims = prims + children_prims

    stage.GetRootLayer().Apply(edits)

    if save_as_path is None:
        omni.usd.get_context().save_stage()
    else:
        omni.usd.get_context().save_as_stage(save_as_path)

def velocity2omega(v_x, v_y, w_z=0, d=0.105, r=0.1):
    omega_0 = (v_x-d*w_z)/r
    omega_1 = -(v_x-math.sqrt(3)*v_y+2*d*w_z)/(2*r)
    omega_2 = -(v_x+math.sqrt(3)*v_y+2*d*w_z)/(2*r)
    return [omega_0, omega_1, omega_2]

class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        self._ext_id = ext_id
        self._extension_path = ext_manager.get_extension_path(ext_id)

        self._menu_items = [
            MenuItemDescription(
                name="Import Robots",
                sub_menu=[
                    MenuItemDescription(name="Mooncake URDF", onclick_fn=lambda a=weakref.proxy(self): a._menu_callback())
                ],
            )
        ]
        add_menu_items(self._menu_items, "Isaac Examples")

        self._build_ui()

    def _build_ui(self):
        self._window = omni.ui.Window(
            EXTENSION_NAME, width=0, height=0, visible=False, dockPreference=ui.DockPreference.LEFT_BOTTOM
        )
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):

                title = "Import a Mooncake Robot via URDF"
                doc_link = "https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_urdf.html"

                overview = "This Example shows you import an NVIDIA Mooncake robot via URDF.\n\nPress the 'Open in IDE' button to view the source code."

                setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)

                frame = ui.CollapsableFrame(
                    title="Command Panel",
                    height=0,
                    collapsed=False,
                    style=get_style(),
                    style_type_name_override="CollapsableFrame",
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                )
                with frame:
                    with ui.VStack(style=get_style(), spacing=5):
                        dict = {
                            "label": "Load Robot",
                            "type": "button",
                            "text": "Load",
                            "tooltip": "Load a Mooncake Robot into the Scene",
                            "on_clicked_fn": self._on_load_robot,
                        }
                        btn_builder(**dict)

                        dict = {
                            "label": "Configure Drives",
                            "type": "button",
                            "text": "Configure",
                            "tooltip": "Configure Joint Drives",
                            "on_clicked_fn": self._on_config_robot,
                        }
                        btn_builder(**dict)

                        dict = {
                            "label": "Spin Robot",
                            "type": "button",
                            "text": "move",
                            "tooltip": "Spin the Robot in Place",
                            "on_clicked_fn": self._on_config_drives,
                        }
                        btn_builder(**dict)

    def on_shutdown(self):
        remove_menu_items(self._menu_items, "Isaac Examples")
        self._window = None

    def _menu_callback(self):
        self._window.visible = not self._window.visible

    def _on_load_robot(self):
        load_stage = asyncio.ensure_future(omni.usd.get_context().new_stage_async())
        asyncio.ensure_future(self._load_mooncake(load_stage))

    async def _load_mooncake(self, task):
        done, pending = await asyncio.wait({task})
        if task in done:
            viewport = omni.kit.viewport_legacy.get_default_viewport_window()
            viewport.set_camera_position("/OmniverseKit_Persp", -1.02, 1.26, 0.5, True)
            viewport.set_camera_target("/OmniverseKit_Persp", 2.20, -2.18, -1.60, True)
            stage = omni.usd.get_context().get_stage()
            scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            result, plane_path = omni.kit.commands.execute(
                "AddGroundPlaneCommand",
                stage=stage,
                planePath="/groundPlane",
                axis="Z",
                size=1500.0,
                position=Gf.Vec3f(0, 0, 0),
                color=Gf.Vec3f(0.5),
            )
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            import_config.merge_fixed_joints = True
            import_config.import_inertia_tensor = False
            # import_config.distance_scale = 100
            import_config.fix_base = False
            import_config.set_make_instanceable(True)
            # import_config.set_instanceable_usd_path("./mooncake_instanceable.usd")
            import_config.set_instanceable_usd_path("omniverse://localhost/Library/Robots/mooncake/mooncake_instanceable.usd")
            import_config.set_default_drive_type(2)     # 0=None, 1=position, 2=velocity
            import_config.make_default_prim = True
            import_config.create_physics_scene = True
            result, robot_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=self._extension_path + "/data/urdf/robots/mooncake/urdf/mooncake.urdf",
                import_config=import_config,
                dest_path="omniverse://localhost/Library/Robots/mooncake/mooncake.usd"
            )
            # convert_asset_instanceable(asset_usd_path=,     # USD file path to the current existing USD asset
            #                            source_prim_path=,   # USD prim path of root prim of the asset
            #                            save_as_path=None,   # USD file path for modified USD stage. Defaults to None, will save in same file.
            #                            create_xforms=True)
            # create_parent_xforms(
            #     asset_usd_path='omniverse://localhost/Library/Robots/mooncake.usd',
            #     source_prim_path="/mooncake",
            #     save_as_path='omniverse://localhost/Library/Robots/mooncake_instanceable.usd'
            # )

            # make sure the ground plane is under root prim and not robot
            # omni.kit.commands.execute(
            #     "MovePrimCommand", path_from=robot_path, path_to="/mooncake", keep_world_transform=True
            # )

            distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
            distantLight.CreateIntensityAttr(500)

    def _on_config_robot(self):
        stage = omni.usd.get_context().get_stage()
        # Make all rollers spin freely by removing extra drive API
        for wheel_index in range(3):
            for plate_index in range(2):
                for roller_index in range(9):
                    prim_path = "/mooncake/wheel_{}/roller_{}_{}_{}_joint".format(wheel_index, wheel_index, plate_index, roller_index)
                    prim = stage.GetPrimAtPath(prim_path)
                    omni.kit.commands.execute(
                        "UnapplyAPISchemaCommand",
                        api=UsdPhysics.DriveAPI,
                        prim=prim,
                        api_prefix="drive",
                        multiple_api_token="angular",
                    )
        ## Attact IMU sensor ##
        self._is = _isaac_sensor.acquire_imu_sensor_interface()
        self.body_path = "/mooncake/base_plate"
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/sensor",
            parent=self.body_path,
            sensor_period=1 / 500.0,            # 2ms
            translation=Gf.Vec3d(0, 0, 17.15),  # translate to surface of /mooncake/top_plate
            orientation=Gf.Quatd(1, 0, 0, 0),   # (x, y, z, w)
            visualize=True,
        )

        omni.kit.commands.execute('ChangeProperty',
            prop_path=Sdf.Path('/mooncake.xformOp:translate'),
            value=Gf.Vec3d(0.0, 0.0, 0.3162),
            prev=Gf.Vec3d(0.0, 0.0, 0.0))
        # Set Damping & Stiffness
# Position Control: for position controlled joints, set a high stiffness and relatively low or zero damping.
# Velocity Control: for velocity controller joints, set a high damping and zero stiffness.
#         omni.kit.commands.execute('ChangeProperty',
#                                   prop_path=Sdf.Path(
#                                       '/mooncake/base_plate/wheel_0_joint.drive:angular:physics:stiffness'),
#                                   value=0.0,
#                                   prev=10000000.0)
#         omni.kit.commands.execute('ChangeProperty',
#                                   prop_path=Sdf.Path(
#                                       '/mooncake/base_plate/wheel_1_joint.drive:angular:physics:stiffness'),
#                                   value=0.0,
#                                   prev=10000000.0)
#         omni.kit.commands.execute('ChangeProperty',
#                                   prop_path=Sdf.Path('/mooncake/base_plate/wheel_2_joint.drive:angular:physics:stiffness'),
#                                   value=0.0,
#                                   prev=10000000.0)

        ###############
        # Create Ball #
        ###############
        # result, ball_path = omni.kit.commands.execute(
        #     "CreatePrimWithDefaultXform",
        #     prim_type="Sphere",
        #     attributes={'radius':0.12},
        #     select_new_prim=True
        # )
        # omni.kit.commands.execute("MovePrimCommand", path_from='/mooncake/Sphere', path_to='/mooncake/ball')
        # omni.kit.commands.execute('ChangeProperty',
        #                           prop_path=Sdf.Path('/mooncake/ball.xformOp:translate'),
        #                           value=Gf.Vec3d(0.0, 0.0, -0.1962),
        #                           prev=Gf.Vec3d(0.0, 0.0, 0.0))
        # omni.kit.commands.execute('SetRigidBody',
        #                           path=Sdf.Path('/mooncake/ball'),
        #                           approximationShape='convexHull',
        #                           kinematic=False)
        #
        # omni.kit.commands.execute('AddPhysicsComponent',
        #                           usd_prim=get_prim_at_path('/mooncake/ball'),
        #                           component='PhysicsMassAPI')
        # # omni.kit.commands.execute('ApplyAPISchema',
        # #                           api= 'pxr.UsdPhysics.MassAPI',
        # #                           prim=get_prim_at_path('/mooncake/ball'))
        # omni.kit.commands.execute('ChangeProperty',
        #                           prop_path=Sdf.Path('/mooncake/ball.physics:mass'),
        #                           value=4.0,
        #                           prev=0.0)


        ## USE 3 IMUs ##
        # result, sensor = omni.kit.commands.execute(
        #     "IsaacSensorCreateImuSensor",
        #     path="/sensor0",
        #     parent=self.body_path,
        #     sensor_period=1 / 500.0,  # 2ms
        #     offset=Gf.Vec3d(0, 15, 17.15),  # translate to upper surface of /mooncake/top_plate
        #     orientation=Gf.Quatd(1, 0, 0, 0),  # (x, y, z, w)
        #     visualize=True,
        # )
        # result, sensor = omni.kit.commands.execute(
        #     "IsaacSensorCreateImuSensor",
        #     path="/sensor1",
        #     parent=self.body_path,
        #     sensor_period=1 / 500.0,  # 2ms
        #     offset=Gf.Vec3d(15*math.sqrt(3)/2, -15/2, 17.15),  # translate to surface of /mooncake/top_plate
        #     orientation=Gf.Quatd(1, 0, 0, 0),  # (x, y, z, w)
        #     visualize=True,
        # )
        # result, sensor = omni.kit.commands.execute(
        #     "IsaacSensorCreateImuSensor",
        #     path="/sensor2",
        #     parent=self.body_path,
        #     sensor_period=1 / 500.0,  # 2ms
        #     offset=Gf.Vec3d(-15*math.sqrt(3)/2, -15/2, 17.15),  # translate to surface of /mooncake/top_plate
        #     orientation=Gf.Quatd(1, 0, 0, 0),  # (x, y, z, w)
        #     visualize=True,
        # )

    def _on_config_drives(self):
        # self._on_config_robot()  # make sure drives are configured first
        stage = omni.usd.get_context().get_stage()
        # set each axis to spin at a rate of 1 rad/s
        axle_0 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/mooncake/base_plate/wheel_0_joint"), "angular")
        axle_1 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/mooncake/base_plate/wheel_1_joint"), "angular")
        axle_2 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/mooncake/base_plate/wheel_2_joint"), "angular")

        omega = velocity2omega(0, 0.1, 0)
        print(omega)
        set_drive_parameters(axle_0, "velocity", math.degrees(omega[0]), 0, math.radians(1e7))
        set_drive_parameters(axle_1, "velocity", math.degrees(omega[1]), 0, math.radians(1e7))
        set_drive_parameters(axle_2, "velocity", math.degrees(omega[2]), 0, math.radians(1e7))
