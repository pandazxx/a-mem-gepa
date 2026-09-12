{
  description = "a-mem-gepa devshell: non-Python deps for optimizing A-MEM prompts with GEPA";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python311
            uv
            just
            git
            sqlite # chromadb's sqlite backend needs a recent sqlite3
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            # libstdc++.so.6 -- uv-installed manylinux wheels (tokenizers,
            # pulled in transitively via litellm) are linked against it, but
            # Nix on Linux doesn't put it on the default dynamic linker
            # search path the way a normal FHS distro does. Not needed on
            # Darwin (different dynamic linker entirely).
            stdenv.cc.cc.lib
          ];

          shellHook = ''
            export UV_PYTHON=${pkgs.python311}/bin/python3.11
          '' + pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          '' + ''
            echo "a-mem-gepa devshell ready. Run 'uv sync' then 'just --list'."
          '';
        };
      });
}
