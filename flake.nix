{
  description = "Tengen development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      qt = pkgs.qt6;
      opencv5 = pkgs.callPackage ./lib/nix/opencv5.nix { };

      # Release builds use -flto, so static libraries hold GCC IR. Plain ar only indexes it through the LTO plugin in
      # binutils' bfd-plugins folder, which other distros provide and Nix does not. gcc-ar loads the plugin itself.
      # Picked up by CMake through the CMAKE_TOOLCHAIN_FILE environment variable, when configuring a new build folder.
      toolchainFile = pkgs.writeText "tengen-nix-toolchain.cmake" ''
        set(CMAKE_AR "${pkgs.stdenv.cc.cc}/bin/gcc-ar" CACHE FILEPATH "")
        set(CMAKE_RANLIB "${pkgs.stdenv.cc.cc}/bin/gcc-ranlib" CACHE FILEPATH "")
      '';

      # Qt loads platform (wayland/xcb) and multimedia plugins at runtime. Unwrapped dev builds need the paths.
      qtPlugins = pkgs.lib.makeSearchPath qt.qtbase.qtPluginPrefix [
        qt.qtbase
        qt.qtwayland
        qt.qtmultimedia
      ];
    in
    {
      packages.${system} = {
        inherit opencv5;
        default = opencv5;
      };

      # Usage: nix develop, then the Linux64-* CMake presets as usual.
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          pkgs.git-lfs
          pkgs.clang-tools # clang-tidy and clangd. CI pins its own clang-format version.
          pkgs.python3 # run-clang-tidy in scripts/tidy.sh.
          pkgs.reuse

          opencv5
          qt.qtbase
          qt.qtmultimedia
          qt.qtwayland
        ];

        shellHook = ''
          export CMAKE_TOOLCHAIN_FILE="${toolchainFile}"
          export QT_PLUGIN_PATH="${qtPlugins}''${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}"
        '';
      };
    };
}
