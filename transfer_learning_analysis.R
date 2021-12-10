library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)


files = dir("all_final_results2/", pattern=".*gz", full.names=TRUE)

files %>%
    purrr::map(function(fn) mutate(read_csv(file=fn, progress=FALSE, col_types = cols()), filename=fn)) %>%
    bind_rows -> df

df = mutate(
    df,
    #design=paste0(design, ifelse(grepl('estimatorTrue', filename), '-weighted', '-unweighted')),
    dgp=ifelse(grepl('results_linear', filename), "Linear", "Nonlinear")
)

mns = df %>%
    group_by(dgp, design, sample_size, x_label, x_value) %>%
    pivot_wider(names_from=metric, values_from=value) %>%
    summarize(
        time_design=mean(time_design + time_estimation),
        bias_ate=mean(ATEError),
        mse_ate=mean(ATEError^2),
        var_ate=var(ATEError)
    ) %>% ungroup() %>%
    pivot_longer(!one_of("dgp", "design", "sample_size", "x_label", "x_value"), names_to="metric")

sems = df %>%
    group_by(dgp, design, sample_size, x_label, x_value) %>%
    pivot_wider(names_from=metric) %>%
    summarize(
        time_design=sd(time_design + time_estimation) / sqrt(n()),
        bias_ate=sd(ATEError) / sqrt(n()),
        mse_ate=sd(ATEError^2) / sqrt(n()),
        var_ate=sd((ATEError-mean(ATEError))^2) / sqrt(n())
    ) %>% ungroup() %>%
    pivot_longer(!one_of("dgp", "design", "sample_size","x_label", "x_value"), names_to="metric", values_to="sem")


log10p1_trans = function() {scales::trans_new(
    'log10p1',
    transform=function(x) log10(x+1),
    inverse=function(x) (10^x) - 1
)}

pdat = left_join(mns, sems, by=c("dgp", "design", "sample_size", "metric", "x_label", "x_value")) %>%
    ungroup() %>%
    mutate(
        n_for_norm=case_when(
            metric=="bias_ate"~sample_size,
            # metric=="mise_ite" & dgp %in% c("QuickBlockDGP", "TwoCircles")~sqrt(sample_size),
            # metric=="mise_ite" & dgp %in% c("LinearDGP", "SinusoidalDGP")~(sample_size)^0.25,
            TRUE~sample_size
        ),
        metric=case_when(
            metric=="bias_ate"~"Bias(ATE)",
            metric=="var_ate"~"Var(ATE)",
            metric=="mse_ate"~"MSE(ATE)",
            TRUE~metric
        )
    )

p_dist = pdat %>%
    #filter(metric=="BIAS(ATE) × n") %>%
    filter(x_label=="Distance", metric != "time_design") %>%
    mutate(
        weighting=factor(ifelse(grepl("weighted-estimatorTrue", design), "Weighted", "Unweighted")),
        weighting=factor(weighting, levels=levels(weighting)[c(2,1)]),
        metric=factor(metric, levels=unique(metric)[c(1,3,2)]),
        balance=case_when(
            grepl("Balance-Source", design)~"Source Balance",
            grepl("Balance-Target", design)~"Target Balance",
            grepl("No-Balance", design)~"Complete Randomization",
        )
    ) %>%
    ggplot(aes(x = x_value, y=abs(value), group=design)) +
    geom_ribbon(aes(fill=balance, ymin=value-1.96*sem, ymax=(value +1.96*sem)), alpha=0.25) +
    geom_line(aes(color=balance, linetype=weighting)) +
    #geom_pointrange(aes(color=design, ymin=abs(value)-sem , ymax=abs(value) +sem)) +
    scale_x_continuous("Distance between the source and the target distribution (δ)") +
    scale_y_continuous("", trans="log10p1", breaks=c(0, 1, 10, 100, 1000)) +
    scale_fill_brewer("", palette="Dark2") +
    scale_color_brewer("", palette="Dark2") +
    scale_linetype_discrete("") +
    facet_wrap(dgp~metric, scales='free', ncol=3) +
    theme_minimal() +
    theme(
        legend.position='bottom',
        panel.border=element_rect(color='black', size=0.5, fill=NA),
        axis.title.x=element_text(size=14),
        axis.title.y=element_text(size=14),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        strip.text.x=element_text(size=12),
    )

print(p_dist)
ggsave("figures/distance.pdf", width = 14, height = 8, device=cairo_pdf)

p_samples = pdat %>%
    #filter(metric=="BIAS(ATE) × n") %>%
    filter(x_label=="Sample Size", metric != "time_design") %>%
    mutate(
        weighting=factor(ifelse(grepl("weighted-estimatorTrue", design), "Weighted", "Unweighted")),
        weighting=factor(weighting, levels=levels(weighting)[c(2,1)]),
        metric=factor(metric, levels=unique(metric)[c(1,3,2)]),
        balance=case_when(
            grepl("Balance-Source", design)~"Source Balance",
            grepl("Balance-Target", design)~"Target Balance",
            grepl("No-Balance", design)~"Complete Randomization",
        )
    ) %>%
    ggplot(aes(x = x_value, y=abs(value), group=design)) +
    geom_ribbon(aes(fill=balance, ymin=value-1.96*sem, ymax=(value +1.96*sem)), alpha=0.25) +
    geom_line(aes(color=balance, linetype=weighting)) +
    #geom_pointrange(aes(color=design, ymin=abs(value)-sem , ymax=abs(value) +sem)) +
    scale_x_continuous("Sample Size") +
    scale_y_continuous("", trans="log10p1") +
    scale_fill_brewer("", palette="Dark2") +
    scale_color_brewer("", palette="Dark2") +
    scale_linetype_discrete("") +
    facet_wrap(dgp~metric, scales='free', ncol=3) +
    theme_minimal() +
    theme(
        legend.position='bottom',
        panel.border=element_rect(color='black', size=0.5, fill=NA),
        axis.title.x=element_text(size=14),
        axis.title.y=element_text(size=14),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        strip.text.x=element_text(size=12),
    )

print(p_samples)
ggsave("figures/samples.pdf", width = 14, height = 8, device=cairo_pdf)


p_thresh = pdat %>%
    #filter(metric=="BIAS(ATE) × n") %>%
    filter(x_label=="Weight Threshold", metric != "time_design") %>%
    mutate(
        weighting=factor(ifelse(grepl("weighted-estimatorTrue", design), "Weighted", "Unweighted")),
        weighting=factor(weighting, levels=levels(weighting)[c(2,1)]),
        metric=factor(metric, levels=unique(metric)[c(1,3,2)]),
        balance=case_when(
            grepl("Balance-Source", design)~"Source Balance",
            grepl("Balance-Target", design)~"Target Balance",
            grepl("No-Balance", design)~"Complete Randomization",
        )
    ) %>%
    ggplot(aes(x = x_value, y=abs(value), group=design)) +
    geom_ribbon(aes(fill=balance, ymin=value-1.96*sem, ymax=(value +1.96*sem)), alpha=0.25) +
    geom_line(aes(color=balance, linetype=weighting)) +
    #geom_pointrange(aes(color=design, ymin=abs(value)-sem , ymax=abs(value) +sem)) +
    scale_x_continuous("Weight Threshhold") +
    scale_y_continuous("", trans="log10p1") +
    scale_fill_brewer("", palette="Dark2") +
    scale_color_brewer("", palette="Dark2") +
    scale_linetype_discrete("") +
    facet_wrap(dgp~metric, scales='free', ncol=3) +
    theme_minimal() +
    theme(
        legend.position='bottom',
        panel.border=element_rect(color='black', size=0.5, fill=NA),
        axis.title.x=element_text(size=14),
        axis.title.y=element_text(size=14),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        strip.text.x=element_text(size=12),
    )

print(p_thresh)
ggsave("figures/threshhold.pdf", width = 14, height = 8, device=cairo_pdf)
