{
  description = "SOMA Retargeter development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems =
        function: nixpkgs.lib.genAttrs systems (system: function (import nixpkgs { inherit system; }));
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          python = pkgs.python312.withPackages (pythonPackages: [ pythonPackages.tkinter ]);
          runtimeLibraries = with pkgs; [
            libGL
            libx11
            libxext
            libxi
            libxrandr
            libxinerama
            libxcursor
            libxkbcommon
            wayland
            zlib
            stdenv.cc.cc.lib
          ];
          mkDevelopmentShell =
            uvExtras:
            pkgs.mkShell {
              packages =
                with pkgs;
                [
                  python
                  uv
                  git
                  git-lfs
                  ffmpeg-headless
                  pkg-config
                  cmake
                  ninja
                ]
                ++ runtimeLibraries;

              UV_PYTHON_DOWNLOADS = "never";
              UV_PROJECT_ENVIRONMENT = ".venv";
              UV_NO_PROGRESS = "1";
              PYTHONPATH = "${python}/${python.sitePackages}";
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibraries;

              shellHook = ''
                export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                if [[ -f pyproject.toml && -f uv.lock ]]; then
                  uv sync --frozen --python "${python}/bin/python" ${uvExtras}
                else
                  echo "SOMA Retargeter shell: enter the repository to sync its uv environment"
                fi
                echo "For Bello, export BELLO_MJCF_PATH=/path/to/bello_full_body_viewer.xml"
              '';
            };
        in
        {
          default = mkDevelopmentShell "";
          amass = mkDevelopmentShell "--extra amass";
        }
      );

      formatter = forAllSystems (pkgs: pkgs.nixfmt);
    };
}
