#Lista de paquetes necesarios
pkgs <- c(
  "jsonlite", "arrow", "dplyr", 
  "tidyr", "lubridate", "stringr", 
  "forecast"
)


installed <- rownames(installed.packages())
to_install <- setdiff(pkgs, installed)

#Instalar sólo los que faltan
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}

# Cargar todos los paquetes
lapply(pkgs, library, character.only = TRUE)

#forecast single hour for a zone, caching the fit + last timestamp
predict_demand_at_time <- function(
    zone_id,
    forecast_time,
    json_file    = "parametersOptimal_updated.json",
    data_folder  = "learningZoneData",
    model_folder = "models",
    force_refit  = FALSE
) {
  #ensure model folder exists
  if (!dir.exists(model_folder)) dir.create(model_folder, recursive = TRUE)
  cache_file <- file.path(model_folder, paste0("zone_", zone_id, "_cache.rds"))
  
  #load JSON params
  params <- fromJSON(json_file, simplifyVector = FALSE)
  matches <- Filter(function(x) x$zone == as.integer(zone_id), params)
  if (length(matches) == 0) {
    #zone not in JSON → return a big negative number
    return(-1e9)
  }
  pz <- matches[[1]]
  pat_seasonal <- "ARIMA\\((\\d+),(\\d+),(\\d+)\\)\\((\\d+),(\\d+),(\\d+)\\)\\[(\\d+)\\]"
  strp <- str_match(pz$model, pat_seasonal)
  
  if (all(is.na(strp))) {
    #no seasonal block → non-seasonal ARIMA(p,d,q)
    pat_noseason <- "ARIMA\\((\\d+),(\\d+),(\\d+)\\)"
    strp2 <- str_match(pz$model, pat_noseason)
    if (any(is.na(strp2))) 
      stop("Invalid model format in JSON.")
    
    p <- as.integer(strp2[2])
    d <- as.integer(strp2[3])
    q <- as.integer(strp2[4])
    #force seasonal part to zero
    P <- D <- Q <- 0
    #assume hourly data default seasonal period
    s <- 24
  } else {
    #the original seasonal case
    p <- as.integer(strp[2]); d <- as.integer(strp[3]); q <- as.integer(strp[4])
    P <- as.integer(strp[5]); D <- as.integer(strp[6]); Q <- as.integer(strp[7])
    s <- as.integer(strp[8])
  }
  
  #load or fit
  if (!force_refit && file.exists(cache_file)) {
    cache <- readRDS(cache_file)
    fit          <- cache$fit
    last_observed <- cache$last_observed
  } else {
    #read & aggregate
    df <- read_parquet(file.path(data_folder, paste0("zone_", zone_id, ".parquet")))
    df <- df %>%
      mutate(hr = floor_date(as.POSIXct(tpep_pickup_datetime, tz = "UTC"), "hour")) %>%
      count(hr, name = "count")
    
    #fill zeros
    all_hours <- tibble(hr = seq(min(df$hr), max(df$hr), by = "hour"))
    df_full   <- all_hours %>% left_join(df, by = "hr") %>% replace_na(list(count = 0))
    
    #ts object
    ts_data <- ts(
      df_full$count,
      frequency = s,
      start     = c(year(df_full$hr[1]), yday(df_full$hr[1]), hour(df_full$hr[1]))
    )
    
    #fit
    fit <- Arima(
      ts_data,
      order    = c(p, d, q),
      seasonal = list(order = c(P, D, Q), period = s),
      include.mean = str_detect(pz$model, "non-zero mean")
    )
    
    #record last observed POSIXct
    last_observed <- max(df_full$hr)
    
    saveRDS(
      list(fit = fit, last_observed = last_observed),
      cache_file
    )
  }
  
  #parse forecast_time explicitly
  ft <- as.POSIXct(forecast_time,
                   format = "%Y-%m-%d %H:%M", 
                   tz     = "UTC")  
  if (is.na(ft)) stop("forecast_time not in 'YYYY-MM-DD HH:MM' format.")
  if (ft <= last_observed) stop("forecast_time must be after last observed hour.")
  h <- as.numeric(difftime(ft, last_observed, units = "hours"))
  
  #forecast
  fc <- forecast(fit, h = h)
  as.numeric(fc$mean[h])
}