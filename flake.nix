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
          ];

          shellHook = ''
            export UV_PYTHON=${pkgs.python311}/bin/python3.11
            echo "a-mem-gepa devshell ready. Run 'uv sync' then 'just --list'."
          '';
        };
      });
}
