# ============================================================
#  Telco Customer Churn — Analyse, Nettoyage & Préparation
#  Fichier R  |  Exécuter EN PREMIER avant le script Python
#  Inclut : équilibrage des classes (ROSE)
# ============================================================

# ── Installer les packages si nécessaire ──────────────────────
# install.packages(c("tidyverse", "caret", "ggplot2", "corrplot", "dplyr", "ROSE"))

library(tidyverse)
library(caret)
library(ggplot2)
library(corrplot)
library(dplyr)
library(ROSE)   # Pour l'équilibrage des classes

# ─────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────
PATH <- "C:/Users/sarra/OneDrive/Bureau/Telco_Churn_Project/"

data <- read.csv(paste0(PATH, "telco_dataset.csv"), stringsAsFactors = FALSE)

cat("── Dimensions du dataset :", dim(data), "\n")
cat("── Colonnes :", paste(names(data), collapse = ", "), "\n\n")

# ─────────────────────────────────────────────────────────────
# 2. ANALYSE EXPLORATOIRE (EDA)
# ─────────────────────────────────────────────────────────────
cat("=== APERÇU DES DONNÉES ===\n")
str(data)

cat("\n=== STATISTIQUES DESCRIPTIVES ===\n")
print(summary(data))

cat("\n=== VALEURS MANQUANTES PAR COLONNE ===\n")
print(colSums(is.na(data)))

cat("\n=== DISTRIBUTION DE LA VARIABLE CIBLE (Churn) ===\n")
print(table(data$Churn))
cat("Proportions (%):\n")
print(round(prop.table(table(data$Churn)) * 100, 2))
# → Résultat attendu : No=73.46% / Yes=26.54%  ← DÉSÉQUILIBRÉ → besoin d'équilibrage !

# ─────────────────────────────────────────────────────────────
# 3. NETTOYAGE DES DONNÉES
# ─────────────────────────────────────────────────────────────

# TotalCharges : 11 valeurs NA → imputation par la médiane
data$TotalCharges <- as.numeric(trimws(data$TotalCharges))
cat("\nValeurs manquantes TotalCharges après conversion :",
    sum(is.na(data$TotalCharges)), "\n")
data$TotalCharges[is.na(data$TotalCharges)] <-
  median(data$TotalCharges, na.rm = TRUE)
# Supprimer NA restants et doublons
data <- na.omit(data)
data <- distinct(data)

cat("── Dimensions après nettoyage :", dim(data), "\n")

# ─────────────────────────────────────────────────────────────
# 4. SÉLECTION DES FEATURES
# ─────────────────────────────────────────────────────────────
features <- c(
  "tenure", "Contract", "MonthlyCharges", "TotalCharges",
  "gender", "Dependents", "SeniorCitizen", "PhoneService",
  "MultipleLines", "InternetService",
  "Churn"
)
data_sel <- data[, features]

cat("\n=== DONNÉES APRÈS SÉLECTION ===\n")
str(data_sel)

# ─────────────────────────────────────────────────────────────
# 5. ENCODAGE DES VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────────────────────

# Binaire Yes/No → 1/0
for (col in c("Dependents", "PhoneService")) {
  data_sel[[col]] <- ifelse(data_sel[[col]] == "Yes", 1L, 0L)
}

# Churn : Yes → 1 / No → 0
data_sel$Churn <- ifelse(data_sel$Churn == "Yes", 1L, 0L)

# Gender : Male → 1 / Female → 0
data_sel$gender <- ifelse(data_sel$gender == "Male", 1L, 0L)

# SeniorCitizen : déjà 0/1
data_sel$SeniorCitizen <- as.integer(data_sel$SeniorCitizen)

# MultipleLines : "No phone service"=0 | "No"=1 | "Yes"=2
data_sel$MultipleLines <- case_when(
  data_sel$MultipleLines == "No phone service" ~ 0L,
  data_sel$MultipleLines == "No"               ~ 1L,
  data_sel$MultipleLines == "Yes"              ~ 2L
)

# InternetService : "No"=0 | "DSL"=1 | "Fiber optic"=2
data_sel$InternetService <- case_when(
  data_sel$InternetService == "No"          ~ 0L,
  data_sel$InternetService == "DSL"         ~ 1L,
  data_sel$InternetService == "Fiber optic" ~ 2L
)

# Contract : "Month-to-month"=0 | "One year"=1 | "Two year"=2
data_sel$Contract <- case_when(
  data_sel$Contract == "Month-to-month" ~ 0L,
  data_sel$Contract == "One year"       ~ 1L,
  data_sel$Contract == "Two year"       ~ 2L
)

cat("\n=== DONNÉES APRÈS ENCODAGE ===\n")
str(data_sel)
print(head(data_sel, 5))

# ─────────────────────────────────────────────────────────────
# 6. VISUALISATIONS EDA
# ─────────────────────────────────────────────────────────────

# 6.1 Distribution du Churn (AVANT équilibrage)
p_imb <- ggplot(data_sel,
                aes(x = factor(Churn, labels = c("No Churn","Churn")),
                    fill = factor(Churn))) +
  geom_bar(show.legend = FALSE) +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("#2196F3","#F44336")) +
  annotate("text", x = 1.5, y = 4700,
           label = "Déséquilibre : 73.46% vs 26.54% → Équilibrage requis !",
           colour = "darkred", fontface = "bold", size = 4) +
  labs(title = "Distribution du Churn (AVANT équilibrage)",
       x = "Churn", y = "Nombre de clients") +
  theme_minimal(base_size = 13)
print(p_imb)
ggsave(paste0(PATH, "plot_churn_avant_equilibrage.png"), p_imb, dpi = 150)

# 6.2 MonthlyCharges vs Churn
p2 <- ggplot(data_sel,
             aes(x = factor(Churn, labels = c("No","Yes")),
                 y = MonthlyCharges, fill = factor(Churn))) +
  geom_boxplot(show.legend = FALSE) +
  scale_fill_manual(values = c("#64B5F6","#EF9A9A")) +
  labs(title = "MonthlyCharges vs Churn",
       x = "Churn", y = "MonthlyCharges") +
  theme_minimal()
print(p2)
ggsave(paste0(PATH, "plot_monthlycharges_churn.png"), p2, dpi = 150)

# 6.3 Tenure vs Churn
p3 <- ggplot(data_sel,
             aes(x = tenure,
                 fill = factor(Churn, labels = c("No","Yes")))) +
  geom_histogram(bins = 30, position = "dodge", alpha = 0.8) +
  scale_fill_manual(name = "Churn", values = c("#42A5F5","#EF5350")) +
  labs(title = "Distribution Tenure par Churn",
       x = "Tenure (mois)", y = "Count") +
  theme_minimal()
print(p3)
ggsave(paste0(PATH, "plot_tenure_churn.png"), p3, dpi = 150)

# 6.4 Matrice de corrélation
png(paste0(PATH, "plot_correlation_matrix.png"), width = 800, height = 700)
corrplot(cor(data_sel), method = "color", type = "upper",
         tl.cex = 0.85, addCoef.col = "black", number.cex = 0.65,
         col   = colorRampPalette(c("#2196F3","white","#F44336"))(200),
         title = "Matrice de Corrélation", mar = c(0,0,2,0))
dev.off()

# ─────────────────────────────────────────────────────────────
# 7. NORMALISATION Min-Max des variables numériques continues
# ─────────────────────────────────────────────────────────────
normalize <- function(x) (x - min(x)) / (max(x) - min(x))

data_sel$tenure         <- normalize(data_sel$tenure)
data_sel$MonthlyCharges <- normalize(data_sel$MonthlyCharges)
data_sel$TotalCharges   <- normalize(data_sel$TotalCharges)

cat("\n=== DONNÉES APRÈS NORMALISATION (5 premières lignes) ===\n")
print(head(data_sel, 5))

# ─────────────────────────────────────────────────────────────
# 8. DIVISION TRAIN / TEST  (80% / 20%)
#    ⚠️  Split AVANT équilibrage — le test reste original !
# ─────────────────────────────────────────────────────────────
set.seed(42)
train_idx  <- createDataPartition(data_sel$Churn, p = 0.80, list = FALSE)
train_data <- data_sel[ train_idx, ]
test_data  <- data_sel[-train_idx, ]

cat("\n── Taille Train (avant équilibrage) :", nrow(train_data), "\n")
cat("── Taille Test  (jamais équilibré)  :", nrow(test_data),  "\n")
cat("\nDistribution Churn — Train (AVANT équilibrage) :\n")
print(table(train_data$Churn))
cat("\nDistribution Churn — Test (dataset réel) :\n")
print(table(test_data$Churn))

# ─────────────────────────────────────────────────────────────
# 9. ÉQUILIBRAGE DES CLASSES SUR LE TRAIN — technique ROSE
#
#    Pourquoi ROSE ?
#    Le dataset est déséquilibré : No=73.46% / Yes=26.54%
#    Sans équilibrage, les modèles ignorent la classe minoritaire
#    ROSE génère des observations synthétiques (sur- et sous-sampling)
#    ⚠️  On n'équilibre JAMAIS le jeu de test (biais d'évaluation)
# ─────────────────────────────────────────────────────────────
cat("\n=== ÉQUILIBRAGE DES CLASSES AVEC ROSE (Train uniquement) ===\n")

# Churn doit être factor pour ROSE
train_data$Churn <- as.factor(train_data$Churn)

set.seed(42)
train_balanced <- ROSE(Churn ~ ., data = train_data, seed = 42)$data

cat("── Taille Train (après équilibrage) :", nrow(train_balanced), "\n")
cat("\nDistribution Churn — Train (APRÈS équilibrage) :\n")
print(table(train_balanced$Churn))
cat("Proportions (%) :\n")
print(round(prop.table(table(train_balanced$Churn)) * 100, 2))

# Remettre Churn en entier pour la sauvegarde CSV
train_balanced$Churn <- as.integer(as.character(train_balanced$Churn))
test_data$Churn      <- as.integer(test_data$Churn)

# Visualisation : AVANT vs APRÈS équilibrage
df_compare <- data.frame(
  Classe  = rep(c("No Churn (0)", "Churn (1)"), 2),
  Periode = c(rep("AVANT", 2), rep("APRÈS", 2)),
  Count   = c(
    as.integer(table(as.factor(train_data$Churn))),
    as.integer(table(as.factor(train_balanced$Churn)))
  )
)

p_bal <- ggplot(df_compare,
                aes(x = Classe, y = Count, fill = Periode)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = Count),
            position = position_dodge(0.9), vjust = -0.4, size = 4) +
  scale_fill_manual(values = c("#4CAF50","#F44336")) +
  labs(title = "Équilibrage des classes — Train uniquement",
       subtitle = "Technique : ROSE (Random Over-Sampling Examples)",
       x = "Classe", y = "Nombre d'observations") +
  theme_minimal(base_size = 12)
print(p_bal)
ggsave(paste0(PATH, "plot_equilibrage_classes.png"), p_bal, dpi = 150)

# ─────────────────────────────────────────────────────────────
# 10. SAUVEGARDE POUR PYTHON
# ─────────────────────────────────────────────────────────────
write.csv(train_balanced, paste0(PATH, "train_data.csv"), row.names = FALSE)
write.csv(test_data,      paste0(PATH, "test_data.csv"),  row.names = FALSE)

cat("\n✅  Fichiers sauvegardés avec succès :\n")
cat("    → train_data.csv  (", nrow(train_balanced),
    "lignes — équilibré avec ROSE)\n")
cat("    → test_data.csv   (", nrow(test_data),
    "lignes — dataset réel, non modifié)\n")
cat("\n➡️   Lancez maintenant le fichier Python pour l'entraînement.\n")