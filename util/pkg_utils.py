from __future__ import absolute_import, division, print_function

import json
import os
import sys

import libtbx.load_env

try:
    import conda
    import pkg_resources
except ImportError:
    conda = None
    pkg_resource = None


def _notice(*lines, **context):
    print(
        os.linesep
        + "=" * 80
        + os.linesep
        + os.linesep
        + os.linesep.join(l.format(**context) for l in lines)
        + os.linesep
        + os.linesep
        + "=" * 80
        + os.linesep
    )


def require_conda_packages(requirements):
    """Ensure DIALS package requirements are met"""
    # Check we can do anything here
    if not libtbx.env.build_options.use_conda:
        return

    if not conda:
        _notice(
            "  WARNING: Can not find conda package in your environment",
            "  You will have to keep track of dependencies yourself",
        )
        return

    conda_list, error, return_code = conda.cli.python_api.run_command(
        conda.cli.python_api.Commands.LIST, "--json", use_exception_handler=True,
    )
    if error or return_code:
        _notice(
            "  WARNING: Could not obtain list of conda packages in your environment",
            error,
        )
        return
    conda_environment = {
        package["name"]: package["version"] for package in json.loads(conda_list)
    }

    action_list = []
    for requirement in requirements:
        requirement = pkg_resources.Requirement.parse(requirement)

        # Check if package is installed in development mode
        if pkg_resources:
            try:
                currentversion = pkg_resources.require(requirement.name)[0].version
            except Exception:
                pass
            else:
                location = None
                for path_item in sys.path:
                    egg_link = os.path.join(path_item, requirement.name + ".egg-link")
                    if os.path.isfile(egg_link):
                        with open(egg_link, "r") as fh:
                            location = fh.readline().strip()
                            break
                if location and currentversion in requirement:
                    print(
                        "requires conda package %s, has %s as developer installation"
                        % (requirement, currentversion)
                    )
                    continue
                elif location and currentversion not in requirement:
                    _notice(
                        "    WARNING: Can not update package {package} automatically.",
                        "",
                        "It is installed as editable package for development purposes. The currently",
                        "installed version, {currentversion}, is too old. The required version is {requirement}.",
                        "Please update the package manually in its installed location:",
                        "",
                        "    {location}",
                        package=requirement.name,
                        currentversion=currentversion,
                        requirement=requirement,
                        location=location,
                    )
                    continue

        # Check if package is installed with conda
        if requirement.name in conda_environment:
            if conda_environment[requirement.name] in requirement:
                print(
                    "requires conda package %s, has %s"
                    % (requirement, conda_environment[requirement.name])
                )
                continue
            print(
                "conda requirement %s is not currently met, current version %s"
                % (requirement, conda_environment[requirement.name])
            )
        else:
            print(
                "conda requirement %s is not currently met, package not installed"
                % (requirement)
            )
            # Install/update required
        action_list.append(str(requirement))

    if not action_list:
        print("All conda requirements satisfied")
        return

    if not os.path.isdir(libtbx.env.under_base(".")):
        _notice(
            "    WARNING: Can not update conda packages automatically.",
            "",
            "You are running in a base-less installation, which disables automatic package",
            "management by convention, see https://github.com/cctbx/cctbx_project/issues/151",
            "",
            "Please update the following packages manually:",
            "  {action_list}",
            action_list=", ".join(action_list),
        )
        return

    if os.getenv("LIBTBX_DISABLE_UPDATES") and os.getenv(
        "LIBTBX_DISABLE_UPDATES"
    ).strip() not in ("0", ""):
        _notice(
            "    WARNING: Can not automatically update conda environment",
            "",
            "Environment variable LIBTBX_DISABLE_UPDATES is set.",
            "Please update the following packages manually:",
            "  {action_list}",
            action_list=", ".join(action_list),
        )
        return

    print(
        "\nUpdating conda environment for packages:"
        + "".join("\n - " + a for a in action_list)
        + "\n"
    )
    _, _, return_code = conda.cli.python_api.run_command(
        conda.cli.python_api.Commands.INSTALL,
        *action_list,
        stdout=None,
        stderr=None,
        use_exception_handler=True
    )
    if return_code:
        _notice(
            "    WARNING: Could not automatically update conda environment",
            "",
            "Please check your environment manually.",
        )
