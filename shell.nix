let
  # We pin to a specific nixpkgs commit for reproducibility.
  # Last updated: 2024-09-03. Check for new commits at https://status.nixos.org.
  pkgs = import (fetchTarball
    "https://github.com/NixOS/nixpkgs/archive/6e99f2a27d60.tar.gz") { };
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      # select Python packages here
    python-pkgs.numpy
    python-pkgs.pandas
    python-pkgs.geopandas
    python-pkgs.scipy
    python-pkgs.seaborn
    python-pkgs.matplotlib
    python-pkgs.notebook
    python-pkgs.scikit-learn
    ]))
  ];
}
