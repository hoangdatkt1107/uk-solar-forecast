// GridSight UK — Azure infrastructure (Container Apps).
//
// Data lake stays on HuggingFace. This deploys ONLY compute + serving:
//   * user-assigned managed identity (pulls images, reads Key Vault)
//   * Key Vault holding the HF token
//   * Storage account + Azure Files share for the forecast JSON handoff
//   * Container Apps environment (+ the Azure Files storage link)
//   * API Container App (always-on, reads the share)
//   * serve Container Apps Job (cron: hourly) — writes the share
//   * retrain Container Apps Job (cron: weekly) — optional
//
// The container registry is created out-of-band (az acr create) so images can be
// pushed BEFORE this template references them; pass its name as `acrName`.

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Short prefix for resource names (lowercase, 3-11 chars).')
@minLength(3)
@maxLength(11)
param namePrefix string = 'gridsight'

@description('Name of the existing Azure Container Registry (created by the deploy script).')
param acrName string

@description('Container image for the API (e.g. myacr.azurecr.io/gridsight-api:latest).')
param apiImage string

@description('Container image for the serve/retrain job.')
param jobImage string

@description('HuggingFace dataset repo id holding the bronze parquet.')
param bronzeHfRepo string

@description('HuggingFace token — stored in Key Vault, never in image or env plaintext.')
@secure()
param hfToken string

@description('Min API replicas. 0 = scale-to-zero (near-$0 when idle, ~seconds cold start on first hit); 1 = always-on (instant, ~few $/mo).')
@minValue(0)
@maxValue(5)
param apiMinReplicas int = 0

@description('Cron for the hourly serve job (UTC). Default: top of every hour.')
param serveCron string = '0 * * * *'

@description('Cron for the weekly retrain job (UTC). Default: Sundays 03:00.')
param retrainCron string = '0 3 * * 0'

@description('Deploy the weekly retrain job too.')
param enableRetrain bool = true

@description('Allowed CORS origins for the API (comma-separated). "*" is fine for public read-only.')
param corsOrigins string = '*'

var suffix = uniqueString(resourceGroup().id)
var kvName = take('${namePrefix}kv${suffix}', 24)
var storageName = take('${namePrefix}st${suffix}', 24)
var shareName = 'serve'
var envStorageName = 'servefiles'
var mountPath = '/mnt/serve'

// ---------------------------------------------------------------- identity ---
resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${namePrefix}-id'
  location: location
}

// ---------------------------------------------------------------- registry ---
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

// AcrPull so the apps/jobs can pull images with the managed identity (no admin creds).
resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, uami.id, 'acrpull')
  scope: acr
  properties: {
    // AcrPull
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ---------------------------------------------------------------- key vault ---
resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: kvName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
  }
}

resource hfSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: kv
  name: 'hf-token'
  properties: {
    value: hfToken
  }
}

// Key Vault Secrets User so the identity can read the HF token at runtime.
resource kvSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(kv.id, uami.id, 'kvsecrets')
  scope: kv
  properties: {
    // Key Vault Secrets User
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ------------------------------------------------------------ files handoff ---
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource share 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: shareName
  properties: {
    shareQuota: 5
  }
}

// ------------------------------------------------------ container apps env ---
resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${namePrefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${namePrefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

// Link the Azure Files share into the environment so app + job can mount it.
resource envStorage 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: env
  name: envStorageName
  properties: {
    azureFile: {
      accountName: storage.name
      accountKey: storage.listKeys().keys[0].value
      shareName: shareName
      accessMode: 'ReadWrite'
    }
  }
}

var acrLoginServer = acr.properties.loginServer
var kvSecretUri = 'https://${kv.name}${environment().suffixes.keyvaultDns}/secrets/hf-token'

// ------------------------------------------------------------------ API app ---
resource apiApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${namePrefix}-api'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        corsPolicy: {
          allowedOrigins: split(corsOrigins, ',')
          allowedMethods: [ 'GET' ]
          allowedHeaders: [ '*' ]
        }
      }
      registries: [
        {
          server: acrLoginServer
          identity: uami.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          env: [
            { name: 'GRIDSIGHT_SERVE_DIR', value: mountPath }
            { name: 'GRIDSIGHT_CORS_ORIGINS', value: corsOrigins }
          ]
          volumeMounts: [
            { volumeName: 'serve', mountPath: mountPath }
          ]
        }
      ]
      scale: {
        minReplicas: apiMinReplicas    // 0 = scale-to-zero (cheapest); KEDA HTTP scales up on request
        maxReplicas: 2
      }
      volumes: [
        {
          name: 'serve'
          storageType: 'AzureFile'
          storageName: envStorage.name
        }
      ]
    }
  }
  dependsOn: [ acrPull ]
}

// -------------------------------------------------------------- serve job ---
resource serveJob 'Microsoft.App/jobs@2024-03-01' = {
  name: '${namePrefix}-serve'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    environmentId: env.id
    configuration: {
      triggerType: 'Schedule'
      replicaTimeout: 3600
      replicaRetryLimit: 1
      scheduleTriggerConfig: {
        cronExpression: serveCron
        parallelism: 1
        replicaCompletionCount: 1
      }
      registries: [
        {
          server: acrLoginServer
          identity: uami.id
        }
      ]
      secrets: [
        {
          name: 'hf-token'
          keyVaultUrl: kvSecretUri
          identity: uami.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'serve'
          image: jobImage
          command: [ 'python', 'pipeline.py', 'serve' ]
          resources: {
            // This is a LEGACY Consumption env (workloadProfiles=null) -> hard cap is
            // 2 vCPU / 4 GiB. (Real 8 GiB needs a workload-profiles env.) Prior OOMs ran
            // at only 1 vCPU / 2 GiB, so 4 GiB here is the untested next step up.
            cpu: json('2.0')
            memory: '4.0Gi'
          }
          env: [
            { name: 'GRIDSIGHT_SERVE_DIR', value: mountPath }
            { name: 'GRIDSIGHT_DATA_DIR', value: '/app/data' }
            { name: 'GRIDSIGHT_BRONZE_HF_REPO', value: bronzeHfRepo }
            { name: 'GRIDSIGHT_HF_TOKEN', secretRef: 'hf-token' }
            // stack-only: drop chronos (torch chronos-bolt load is the heaviest step and
            // a nice-to-have) so the hourly run stays comfortably inside 4 GiB.
            { name: 'GRIDSIGHT_MODELS', value: 'stack' }
            // sync only the recent tail from HF, not the full 2yr history (that is a
            // weekly-retrain job). Keeps the hourly run fast + light.
            { name: 'GRIDSIGHT_SYNC_MONTHS', value: '2' }
            // NESO archive is ~38M rows across all years; serve reads only the current
            // year (still far more than the ~1wk of lags it needs) to stay under 4 GiB.
            { name: 'GRIDSIGHT_NESO_LEAN', value: '1' }
            // Skip the Met AWS "overwrite last day" pass: it re-fetches whole-day netCDF
            // and a single corrupt S3 download corrupts the HDF5 native heap (double free)
            // and kills the job. The gap-fill pass before it already covers late leads.
            { name: 'GRIDSIGHT_MET_OVERWRITE', value: '0' }
          ]
          volumeMounts: [
            { volumeName: 'serve', mountPath: mountPath }
          ]
        }
      ]
      volumes: [
        {
          name: 'serve'
          storageType: 'AzureFile'
          storageName: envStorage.name
        }
      ]
    }
  }
  dependsOn: [ acrPull, kvSecretsUser, hfSecret ]
}

// ------------------------------------------------------------ retrain job ---
resource retrainJob 'Microsoft.App/jobs@2024-03-01' = if (enableRetrain) {
  name: '${namePrefix}-retrain'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    environmentId: env.id
    configuration: {
      triggerType: 'Schedule'
      replicaTimeout: 7200
      replicaRetryLimit: 1
      scheduleTriggerConfig: {
        cronExpression: retrainCron
        parallelism: 1
        replicaCompletionCount: 1
      }
      registries: [
        {
          server: acrLoginServer
          identity: uami.id
        }
      ]
      secrets: [
        {
          name: 'hf-token'
          keyVaultUrl: kvSecretUri
          identity: uami.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'retrain'
          image: jobImage
          command: [ 'python', 'pipeline.py', 'retrain' ]
          resources: {
            // Legacy Consumption env cap: 2 vCPU / 4 GiB (same as serve). Training is
            // heavier, so if this OOMs, the fix is a workload-profiles env, not more here.
            cpu: json('2.0')
            memory: '4.0Gi'
          }
          env: [
            { name: 'GRIDSIGHT_SERVE_DIR', value: mountPath }
            { name: 'GRIDSIGHT_DATA_DIR', value: '/app/data' }
            { name: 'GRIDSIGHT_BRONZE_HF_REPO', value: bronzeHfRepo }
            { name: 'GRIDSIGHT_HF_TOKEN', secretRef: 'hf-token' }
          ]
          volumeMounts: [
            { volumeName: 'serve', mountPath: mountPath }
          ]
        }
      ]
      volumes: [
        {
          name: 'serve'
          storageType: 'AzureFile'
          storageName: envStorage.name
        }
      ]
    }
  }
  dependsOn: [ acrPull, kvSecretsUser, hfSecret ]
}

output apiUrl string = 'https://${apiApp.properties.configuration.ingress.fqdn}'
output acrLoginServer string = acrLoginServer
output keyVaultName string = kv.name
output identityClientId string = uami.properties.clientId
