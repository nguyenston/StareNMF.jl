{ pkgs, ... }:

{
  env.GREET = "Entering env...";
  packages = with pkgs; [
    cmdstan
  ];

  enterShell = ''
    echo $GREET
  '';

  languages.python = {
    enable = true;
    version = "3.11";
    uv.enable = true;
    venv.enable = true;
    venv.requirements = ''
      scipy
      autograd
      pandas
      matplotlib
      numpy
    '';
  };
}
