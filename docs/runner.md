# Self-hosted GPU runner

GitHub-hosted runners have no GPU, so `gpu`-marked tests, `slow` backtests and
the full GPU weekly pipeline run on the project workstation (Intel i9-14900,
RTX 4500 Ada 24 GB, 128 GB RAM). The workflow is
[`.github/workflows/gpu.yml`](https://github.com/guru-bharadwaj20/Kairos-Observatory/blob/main/.github/workflows/gpu.yml).

## Labels

The runner is registered with these labels. `gpu.yml` requires all of them.

| Label | Meaning |
|---|---|
| `self-hosted`, `windows`, `x64` | Added by GitHub automatically |
| `gpu` | A CUDA GPU is present |
| `kairos` | This project's workstation, not a shared runner |

## Security

A self-hosted runner executes whatever code a workflow checks out. On a public
repository that must never include code from fork pull requests.

- `gpu.yml` triggers only on `push` to `main`, `schedule` and
  `workflow_dispatch`. It has no `pull_request` trigger.
- In *Settings → Actions → General*, set "Fork pull request workflows" to
  require approval for all outside collaborators.
- Run the runner service as a dedicated local user with no admin rights and no
  access to personal files.
- The job token is read-only (`permissions: contents: read`). Ledger commits
  come from the hosted `weekly.yml` job, not from this machine.

## One-time setup (Windows, PowerShell)

1. Prerequisites: NVIDIA driver with CUDA 12.x, Git, PowerShell 7 (`pwsh`), and
   `uv` on the service user's `PATH`.

   ```powershell
   winget install --id Microsoft.PowerShell
   winget install --id astral-sh.uv
   nvidia-smi   # must list the RTX 4500 Ada
   ```

2. Create a dedicated user, for example `kairos-runner`, and log in as it once.

3. In the repository go to *Settings → Actions → Runners → New self-hosted
   runner*, choose Windows x64, and follow the download and checksum steps into
   `C:\actions-runner`.

4. Configure it with the project labels and install it as a service:

   ```powershell
   cd C:\actions-runner
   .\config.cmd --url https://github.com/guru-bharadwaj20/Kairos-Observatory `
                --token <REGISTRATION_TOKEN> `
                --name kairos-workstation `
                --labels gpu,kairos `
                --runasservice `
                --windowslogonaccount .\kairos-runner
   ```

5. Check that the runner shows as *Idle* with all five labels, then start
   `gpu.yml` from the *Actions* tab with *Run workflow*.

## Resource limits

Jobs follow the hardware plan in the [design notes](design.md#hardware-plan-for-this-workstation): 28 worker
processes at most, BLAS threads pinned to 1, peak VRAM under 20 GB. The
workflow's `concurrency` group keeps it to one GPU job at a time.

## Maintenance

- The runner updates itself. Update the NVIDIA driver by hand and re-run
  `uv run kairos doctor` afterwards.
- To remove it: `.\config.cmd remove --token <REMOVAL_TOKEN>`.
