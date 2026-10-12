{
  description = "Notebook for viewing Claude spend in the claude-box ledger";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3.withPackages (ps: with ps; [
          jupyterlab
          nbconvert
          ipykernel
          polars
          pyarrow
          seaborn
          matplotlib
          pytest
          mypy
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ python pkgs.ruff ];
          shellHook = ''
            export PYTHONPATH="$PWD''${PYTHONPATH:+:$PYTHONPATH}"
          '';
        };
      });
}
