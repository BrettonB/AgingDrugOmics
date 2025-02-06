library(stringr)
library(tibble)

deconvolute_tg <- function(metabolite) {
  # Split the metabolite into whole and FFAs parts
  parts <- str_split(metabolite, "\\|", n = 2, simplify = TRUE)
  whole <- parts[1]
  ffas <- if(length(parts) > 1) parts[2] else NA
  
  # Process the whole part
  whole_parts <- str_match(whole, "(TG O-)?(\\d+):(\\d+)(;(\\d+)O)?")[1,]
  total_c_num <- as.numeric(whole_parts[3])
  saturation_number <- as.numeric(whole_parts[4])
  hertoatoms <- if(!is.na(whole_parts[5])) as.numeric(whole_parts[6]) else 0
  
  # Process the FFAs part
  process_ffa <- function(ffa) {
    parts <- str_match(ffa, "(TG O-)?(\\d+):(\\d+)(;(\\d+)O)?")[1,]
    c_num <- as.numeric(parts[3])
    sat_num <- as.numeric(parts[4])
    het_num <- if(!is.na(parts[5])) as.numeric(parts[6]) else 0
    return(c(c_num, sat_num, het_num))
  }
  
  ffa_list <- str_split(ffas, "_")[[1]]
  ffa_data <- sapply(ffa_list, process_ffa)
  
  # Ensure we have data for all three FFAs
  if(ncol(ffa_data) < 3) {
    ffa_data <- cbind(ffa_data, matrix(NA, nrow = 3, ncol = 3 - ncol(ffa_data)))
  }
  
  # Create the result
  result <- tibble(
    Metabolite = metabolite,
    Total_C_Num = total_c_num,
    Saturation_Number = saturation_number,
    Hertoatoms = hertoatoms,
    TG_One_C_Num = ffa_data[1,1],
    TG_One_Saturation_Number = ffa_data[2,1],
    TG_One_Hertoatoms = ffa_data[3,1],
    TG_Two_C_Num = ffa_data[1,2],
    TG_Two_Saturation_Number = ffa_data[2,2],
    TG_Two_Hertoatoms = ffa_data[3,2],
    TG_Three_C_Num = ffa_data[1,3],
    TG_Three_Saturation_Number = ffa_data[2,3],
    TG_Three_Hertoatoms = ffa_data[3,3]
  )
  
  return(result)
}