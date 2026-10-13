# Thiết lập Google Cloud MỘT LẦN cho web demo trên Cloud Run (chạy trên máy đã cài gcloud và gh).
#   gcloud auth login
#   powershell -ExecutionPolicy Bypass -File deploy/cloudrun/setup_gcp.ps1 -ProjectId <id-project>
# Trước đó: tạo project trên console.cloud.google.com và liên kết tài khoản thanh toán (Billing).
# Script tạo: Artifact Registry "dsp-s" (giữ 2 image mới nhất), service account deploy + runtime,
# secret "dsp-demo-pass" (mật khẩu web), Workload Identity Federation cho đúng repo GitHub (không cần khóa JSON),
# rồi ghi các biến GCP_* vào GitHub để job deploy-cloudrun chạy. Chạy lại nhiều lần không sao (bỏ qua phần đã có).
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "asia-east1",
    [string]$GithubRepo = "remembered-fragrance/dsp-s-reltr-ag"
)
$ErrorActionPreference = "Continue"

function Run([string[]]$a) {
    Write-Host "> gcloud $($a -join ' ')" -ForegroundColor DarkGray
    & gcloud @a
    if ($LASTEXITCODE -ne 0) { throw "gcloud lỗi (mã $LASTEXITCODE): $($a -join ' ')" }
}
function Exists([string[]]$a) {
    & gcloud @a *> $null
    return ($LASTEXITCODE -eq 0)
}

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { throw "Chưa có gcloud: winget install Google.CloudSDK, mở terminal mới, gcloud auth login" }

Run @("config", "set", "project", $ProjectId)
Run @("services", "enable", "cloudbilling.googleapis.com", "run.googleapis.com", "artifactregistry.googleapis.com",
      "iam.googleapis.com", "iamcredentials.googleapis.com", "sts.googleapis.com", "secretmanager.googleapis.com")
$billing = (& gcloud billing projects describe $ProjectId --format="value(billingEnabled)") 2>$null
if ($billing -ne "True") { throw "Project $ProjectId chưa liên kết tài khoản thanh toán: Console → Billing → Link a billing account" }
$num = (& gcloud projects describe $ProjectId --format="value(projectNumber)").Trim()

# Artifact Registry: chỉ giữ 2 image mới nhất (dung lượng miễn phí 0,5 GB)
if (-not (Exists @("artifacts", "repositories", "describe", "dsp-s", "--location=$Region"))) {
    Run @("artifacts", "repositories", "create", "dsp-s", "--repository-format=docker", "--location=$Region",
          "--description=DSP_S web demo")
}
$policy = Join-Path $env:TEMP "dsp_cleanup.json"
[IO.File]::WriteAllText($policy, '[{"name":"keep-2","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":2}},{"name":"delete-rest","action":{"type":"Delete"},"condition":{"tagState":"any"}}]')
Run @("artifacts", "repositories", "set-cleanup-policies", "dsp-s", "--location=$Region", "--policy=$policy", "--no-dry-run")
Remove-Item $policy

# Service account: gh-deploy (GitHub Actions dùng để deploy), dsp-demo-run (Cloud Run chạy bằng tài khoản này)
$deployer = "gh-deploy@$ProjectId.iam.gserviceaccount.com"
$runtime = "dsp-demo-run@$ProjectId.iam.gserviceaccount.com"
foreach ($sa in @(@("gh-deploy", "GitHub Actions deploy DSP_S"), @("dsp-demo-run", "Cloud Run runtime DSP_S demo"))) {
    if (-not (Exists @("iam", "service-accounts", "describe", "$($sa[0])@$ProjectId.iam.gserviceaccount.com"))) {
        Run @("iam", "service-accounts", "create", $sa[0], "--display-name=$($sa[1])")
    }
}
Run @("projects", "add-iam-policy-binding", $ProjectId, "--member=serviceAccount:$deployer", "--role=roles/run.admin", "--condition=None", "--quiet")
Run @("artifacts", "repositories", "add-iam-policy-binding", "dsp-s", "--location=$Region", "--member=serviceAccount:$deployer",
      "--role=roles/artifactregistry.writer", "--quiet")
Run @("iam", "service-accounts", "add-iam-policy-binding", $runtime, "--member=serviceAccount:$deployer",
      "--role=roles/iam.serviceAccountUser", "--quiet")

# Mật khẩu web demo trong Secret Manager (user đăng nhập: dsp)
if (-not (Exists @("secrets", "describe", "dsp-demo-pass"))) {
    Run @("secrets", "create", "dsp-demo-pass", "--replication-policy=automatic")
}
$sec = Read-Host "Đặt mật khẩu cho web demo (user: dsp, ít nhất 8 ký tự; Enter để giữ mật khẩu cũ)" -AsSecureString
$pw = [System.Net.NetworkCredential]::new('', $sec).Password
if ($pw) {
    if ($pw.Length -lt 8) { throw "Mật khẩu cần ít nhất 8 ký tự" }
    $tmp = Join-Path $env:TEMP ("dsp_" + [guid]::NewGuid().ToString("N"))
    [IO.File]::WriteAllText($tmp, $pw)   # không có xuống dòng ở cuối
    try { Run @("secrets", "versions", "add", "dsp-demo-pass", "--data-file=$tmp") } finally { Remove-Item $tmp -Force }
}
Run @("secrets", "add-iam-policy-binding", "dsp-demo-pass", "--member=serviceAccount:$runtime",
      "--role=roles/secretmanager.secretAccessor", "--quiet")

# Workload Identity Federation: chỉ workflow của repo $GithubRepo được mượn quyền gh-deploy
if (-not (Exists @("iam", "workload-identity-pools", "describe", "github", "--location=global"))) {
    Run @("iam", "workload-identity-pools", "create", "github", "--location=global", "--display-name=GitHub Actions")
}
if (-not (Exists @("iam", "workload-identity-pools", "providers", "describe", "github", "--location=global", "--workload-identity-pool=github"))) {
    Run @("iam", "workload-identity-pools", "providers", "create-oidc", "github", "--location=global",
          "--workload-identity-pool=github", "--display-name=GitHub", "--issuer-uri=https://token.actions.githubusercontent.com",
          "--attribute-mapping=google.subject=assertion.sub,attribute.repository=assertion.repository",
          "--attribute-condition=assertion.repository=='$GithubRepo'")
}
Run @("iam", "service-accounts", "add-iam-policy-binding", $deployer, "--role=roles/iam.workloadIdentityUser",
      "--member=principalSet://iam.googleapis.com/projects/$num/locations/global/workloadIdentityPools/github/attribute.repository/$GithubRepo",
      "--quiet")
$provider = "projects/$num/locations/global/workloadIdentityPools/github/providers/github"

# Biến cho GitHub Actions (không bí mật)
$vars = [ordered]@{ GCP_PROJECT_ID = $ProjectId; GCP_REGION = $Region; GCP_WIF_PROVIDER = $provider; GCP_SA_EMAIL = $deployer; GCP_RUN_SA = $runtime }
if (Get-Command gh -ErrorAction SilentlyContinue) {
    foreach ($k in $vars.Keys) { gh variable set $k -b $vars[$k] -R $GithubRepo }
    Write-Host "Đã ghi biến GCP_* vào GitHub ($GithubRepo)." -ForegroundColor Green
} else {
    Write-Host "Chưa có gh: tự đặt các biến sau ở GitHub → Settings → Secrets and variables → Actions → Variables:"
    $vars.GetEnumerator() | ForEach-Object { Write-Host "  $($_.Key) = $($_.Value)" }
}
Write-Host ""
Write-Host "Xong. Deploy: push lên main, hoặc  gh workflow run ci-cd.yml -R $GithubRepo" -ForegroundColor Green
Write-Host "Nên đặt cảnh báo ngân sách (ví dụ 1 USD): https://console.cloud.google.com/billing/budgets"
