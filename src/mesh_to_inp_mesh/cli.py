import argparse
from pathlib import Path
from rich.console import Console

from mesh_to_inp_mesh.convert import convert

def main():
    parser = argparse.ArgumentParser(
        prog = ".mesh to .inp mesh", 
        description = "Convert a .mesh (or other meshio-supported mesh) to .inp with interfaces",
    )
    parser.add_argument(
        "input_path",
        type = Path,
        help = "Path to the input mesh file",
        )
    parser.add_argument(
        "-o",
        "--out",
        type = Path,
        default = None,
        help = "Path where the output needs to be saved (with file name and suffix)"
    )

    console = Console()


    console.print("[bold]mesh-to-inp-mesh[/bold]")

    args = parser.parse_args()
    out_path = args.out if args.out else args.input_path.with_suffix(".inp")

    with console.status("[cyan]Converting to .inp..."):
        convert(args.input_path, out_path)

    console.print(f"[green]✔ Converting to .inp complete[/green]")
    console.print(f"Wrote: {out_path.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()