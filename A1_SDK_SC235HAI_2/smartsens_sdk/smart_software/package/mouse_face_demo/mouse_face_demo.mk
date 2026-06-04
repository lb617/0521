MOUSE_FACE_DEMO_VERSION =
MOUSE_FACE_DEMO_SITE = $(S1SRC)/app_demo/mouse_face_detection
MOUSE_FACE_DEMO_SITE_METHOD = local

export EXPORT_LIB_M1_SDK_ROOT_PATH = $(call qstrip,$(BR2_M1_SDK_ROOT_PATH))

define MOUSE_FACE_DEMO_BUILD_CMDS
	$(MAKE) CC="$(TARGET_CC)" -C $(@D) all
endef

define MOUSE_FACE_DEMO_INSTALL_TARGET_CMDS
	mkdir -p $(TARGET_DIR)/app_demo/
	$(INSTALL) -D -m 0755 $(@D)/mouse_face_demo $(TARGET_DIR)/app_demo/
	cp -r $(@D)/app_assets/. $(TARGET_DIR)/app_demo/app_assets/
	cp -r $(@D)/scripts/. $(TARGET_DIR)/app_demo/scripts/
	chmod +x $(TARGET_DIR)/app_demo/scripts/run.sh
endef

$(eval $(cmake-package))
